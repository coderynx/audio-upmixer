#import "audio_bridge.h"

#import <AVFoundation/AVFoundation.h>
#import <AVFAudio/AVFAudio.h>
#import <CoreMedia/CoreMedia.h>
#import <Foundation/Foundation.h>
#import <time.h>

static const AVAudioFrameCount BUFFER_FRAMES = 512;
static const NSUInteger BUFFER_COUNT = 4;
static const int64_t MEDIA_PREFILL_FRAMES = 2048;
static const int64_t MEDIA_QUEUE_FRAMES = 16384;

@interface UpmixerAudio : NSObject
@property(nonatomic, strong) AVAudioEngine *engine;
@property(nonatomic, strong) AVAudioPlayerNode *player;
@property(nonatomic, strong) AVSampleBufferAudioRenderer *renderer;
@property(nonatomic, strong) AVSampleBufferRenderSynchronizer *synchronizer;
@property(nonatomic, strong) AVAudioFormat *format;
@property(nonatomic, strong) NSMutableArray *pending;
@property(nonatomic, strong) NSMutableArray<AVAudioPCMBuffer *> *available;
@property(nonatomic) dispatch_semaphore_t semaphore;
@property(nonatomic) dispatch_queue_t queue;
@property(nonatomic, strong) id flushObserver;
@property(nonatomic, copy) NSString *failure;
@property(nonatomic) BOOL requesting;
@property(nonatomic) BOOL playing;
@property(nonatomic) BOOL started;
@property(nonatomic) BOOL finished;
@property(nonatomic) BOOL upmixer714;
@property(nonatomic) int64_t startFrame;
@property(nonatomic) int64_t nextFrame;
@property(nonatomic) int64_t lastFrame;
@property(nonatomic) double lastProgress;
@property(nonatomic) double lastEnqueue;
@end

@implementation UpmixerAudio
@end

static void set_error(char **out, NSError *error, NSString *fallback) {
    if (!out) return;
    *out = strdup((error.localizedDescription ?: fallback).UTF8String);
}

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

static AudioChannelLayoutTag layout_tag(NSString *layout) {
    if ([layout isEqualToString:@"5.1.2"]) return kAudioChannelLayoutTag_Atmos_5_1_2;
    if ([layout isEqualToString:@"5.1.4"]) return kAudioChannelLayoutTag_Atmos_5_1_4;
    if ([layout isEqualToString:@"7.1.2"]) return kAudioChannelLayoutTag_Atmos_7_1_2;
    if ([layout isEqualToString:@"7.1.4"]) return kAudioChannelLayoutTag_Atmos_7_1_4;
    if ([layout isEqualToString:@"5.1"]) return kAudioChannelLayoutTag_MPEG_5_1_A;
    if ([layout isEqualToString:@"7.1"]) return kAudioChannelLayoutTag_MPEG_7_1_A;
    return kAudioChannelLayoutTag_Stereo;
}

bool upmixer_audio_uses_media_pipeline(const char *layout_name, bool spatial) {
    if (!spatial) return false;
    @autoreleasepool {
        NSString *layout = [NSString stringWithUTF8String:layout_name ?: "stereo"];
        return ![layout isEqualToString:@"stereo"];
    }
}

static int64_t media_frame(UpmixerAudio *host) {
    CMTime time = host.synchronizer.currentTime;
    if (!host.started || !CMTIME_IS_NUMERIC(time)) return host.startFrame;
    return MAX(host.startFrame, CMTimeConvertScale(time, 48000,
                         kCMTimeRoundingMethod_RoundTowardZero).value);
}

static BOOL media_ok(UpmixerAudio *host, char **error) {
    if (host.renderer.status == AVQueuedSampleBufferRenderingStatusFailed) {
        set_error(error, host.renderer.error, @"Apple media renderer failed");
        return NO;
    }
    int64_t frame = media_frame(host);
    double now = monotonic_seconds();
    if (!host.playing || frame != host.lastFrame) {
        host.lastFrame = frame;
        host.lastProgress = now;
    }
    if (host.playing && host.nextFrame > frame && now - host.lastProgress > 3.0) {
        host.failure = @"Apple media playback clock stalled; restart preview or check the output device";
    }
    if (host.playing && host.pending.count && now - host.lastEnqueue > 3.0) {
        host.failure = @"Apple media renderer stopped accepting audio; restart preview";
    }
    if (host.failure) {
        set_error(error, nil, host.failure);
        return NO;
    }
    return YES;
}

static void start_media_if_ready(UpmixerAudio *host) {
    if (!host.playing || host.started || host.nextFrame == host.startFrame) return;
    if (!host.finished && host.nextFrame - host.startFrame < MEDIA_PREFILL_FRAMES) return;
    host.started = YES;
    host.lastProgress = host.lastEnqueue = monotonic_seconds();
    [host.synchronizer setRate:1.0f time:CMTimeMake(host.startFrame, 48000)];
}

static void request_media(UpmixerAudio *host) {
    if (host.requesting || host.pending.count == 0) return;
    host.requesting = YES;
    __weak UpmixerAudio *weakHost = host;
    [host.renderer requestMediaDataWhenReadyOnQueue:host.queue usingBlock:^{
        UpmixerAudio *strongHost = weakHost;
        if (!strongHost) return;
        while (strongHost.pending.count && strongHost.renderer.readyForMoreMediaData) {
            CMSampleBufferRef sample = (__bridge CMSampleBufferRef)strongHost.pending.firstObject;
            [strongHost.renderer enqueueSampleBuffer:sample];
            strongHost.lastEnqueue = monotonic_seconds();
            [strongHost.pending removeObjectAtIndex:0];
        }
        if (strongHost.pending.count == 0) {
            [strongHost.renderer stopRequestingMediaData];
            strongHost.requesting = NO;
        }
    }];
}

UpmixerAudioHost upmixer_audio_create(const char *layout_name, bool spatial, bool head_tracking,
                                      int64_t start_frame, char **error) {
    @autoreleasepool {
        (void)head_tracking; // macOS owns media head tracking in Control Center.
        NSString *name = [NSString stringWithUTF8String:layout_name ?: "stereo"];
        NSArray *supported = @[@"stereo", @"5.1", @"7.1", @"5.1.2", @"5.1.4", @"7.1.2", @"7.1.4"];
        if (![supported containsObject:name] || start_frame < 0 || start_frame > INT64_MAX - MEDIA_QUEUE_FRAMES) {
            set_error(error, nil, @"Invalid native audio layout or start frame");
            return NULL;
        }
        AVAudioChannelLayout *layout = [[AVAudioChannelLayout alloc] initWithLayoutTag:layout_tag(name)];
        UpmixerAudio *host = [UpmixerAudio new];
        host.upmixer714 = [name isEqualToString:@"7.1.4"];
        host.startFrame = host.nextFrame = host.lastFrame = start_frame;
        host.lastProgress = host.lastEnqueue = monotonic_seconds();
        BOOL media = upmixer_audio_uses_media_pipeline(layout_name, spatial);
        host.format = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                    sampleRate:48000 interleaved:media channelLayout:layout];
        if (!host.format) {
            set_error(error, nil, @"Could not create the native audio format");
            return NULL;
        }
        if (media) {
            host.queue = dispatch_queue_create("com.coderynx.upmixer.media",
                dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
            host.pending = [NSMutableArray new];
            host.renderer = [AVSampleBufferAudioRenderer new];
            if (!host.renderer) {
                set_error(error, nil, @"Apple media audio service is unavailable");
                return NULL;
            }
            host.renderer.allowedAudioSpatializationFormats = AVAudioSpatializationFormatMultichannel;
            host.renderer.audioTimePitchAlgorithm = AVAudioTimePitchAlgorithmVarispeed;
            host.synchronizer = [AVSampleBufferRenderSynchronizer new];
            // Startup is gated by our bounded prefill, including short clips at EOF.
            host.synchronizer.delaysRateChangeUntilHasSufficientMediaData = NO;
            [host.synchronizer addRenderer:host.renderer];
            [host.synchronizer setRate:0.0f time:CMTimeMake(start_frame, 48000)];
            __weak UpmixerAudio *weakHost = host;
            host.flushObserver = [[NSNotificationCenter defaultCenter]
                addObserverForName:AVSampleBufferAudioRendererWasFlushedAutomaticallyNotification
                object:host.renderer queue:nil usingBlock:^(NSNotification *note) {
                    (void)note;
                    UpmixerAudio *strongHost = weakHost;
                    if (!strongHost) return;
                    dispatch_async(strongHost.queue, ^{
                        strongHost.failure = @"Apple media output was reset; restart preview";
                    });
                }];
        } else {
            host.engine = [AVAudioEngine new];
            host.player = [AVAudioPlayerNode new];
            [host.engine attachNode:host.player];
            [host.engine connect:host.player to:host.engine.outputNode format:host.format];
            host.available = [NSMutableArray arrayWithCapacity:BUFFER_COUNT];
            host.semaphore = dispatch_semaphore_create(BUFFER_COUNT);
            for (NSUInteger i = 0; i < BUFFER_COUNT; i++) {
                [host.available addObject:[[AVAudioPCMBuffer alloc] initWithPCMFormat:host.format
                                                                       frameCapacity:BUFFER_FRAMES]];
            }
        }
        return (__bridge_retained void *)host;
    }
}

void upmixer_audio_set_head_tracking(UpmixerAudioHost opaque, bool enabled) {
    // Retained ABI for saved preview requests; media head tracking is system controlled.
    (void)opaque;
    (void)enabled;
}

bool upmixer_audio_start(UpmixerAudioHost opaque, char **error) {
    @autoreleasepool {
        UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
        if (host.renderer) return true;
        NSError *engineError = nil;
        if (![host.engine startAndReturnError:&engineError]) {
            set_error(error, engineError, @"Could not start direct audio output");
            return false;
        }
        return true;
    }
}

void upmixer_audio_pause(UpmixerAudioHost opaque) {
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (host.renderer) {
        dispatch_sync(host.queue, ^{
            host.playing = NO;
            host.synchronizer.rate = 0.0f;
        });
    } else {
        [host.player pause];
    }
}

void upmixer_audio_resume(UpmixerAudioHost opaque) {
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (host.renderer) {
        dispatch_sync(host.queue, ^{
            host.playing = YES;
            host.lastProgress = host.lastEnqueue = monotonic_seconds();
            if (host.started) host.synchronizer.rate = 1.0f;
            else start_media_if_ready(host);
        });
    } else {
        [host.player play];
    }
}

int upmixer_audio_ready(UpmixerAudioHost opaque, char **error) {
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (!host.renderer) return 1;
    __block int ready;
    dispatch_sync(host.queue, ^{
        ready = media_ok(host, error)
            ? (host.nextFrame - media_frame(host) + BUFFER_FRAMES <= MEDIA_QUEUE_FRAMES
                && host.pending.count < MEDIA_QUEUE_FRAMES / BUFFER_FRAMES) : -1;
    });
    return ready;
}

static uint32_t source_channel(UpmixerAudio *host, uint32_t channel) {
    if (host.upmixer714 && channel >= 4 && channel < 8) {
        return channel < 6 ? channel + 2 : channel - 2;
    }
    return channel;
}

static CMSampleBufferRef media_sample(UpmixerAudio *host, const float *const *channels,
                                      uint32_t frames, char **error) {
    size_t count = host.format.channelCount;
    size_t sampleSize = count * sizeof(float);
    size_t byteCount = frames * sampleSize;
    CMBlockBufferRef block = NULL;
    OSStatus status = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, NULL, byteCount,
        kCFAllocatorDefault, NULL, 0, byteCount, kCMBlockBufferAssureMemoryNowFlag, &block);
    char *bytes = NULL;
    if (status == noErr) status = CMBlockBufferGetDataPointer(block, 0, NULL, NULL, &bytes);
    CMSampleBufferRef sample = NULL;
    if (status == noErr) {
        float *interleaved = (float *)bytes;
        for (uint32_t frame = 0; frame < frames; frame++) {
            for (uint32_t channel = 0; channel < count; channel++) {
                interleaved[frame * count + channel] = channels[source_channel(host, channel)][frame];
            }
        }
        CMSampleTimingInfo timing = {CMTimeMake(1, 48000), CMTimeMake(host.nextFrame, 48000), kCMTimeInvalid};
        status = CMSampleBufferCreateReady(kCFAllocatorDefault, block, host.format.formatDescription,
                                           frames, 1, &timing, 1, &sampleSize, &sample);
    }
    if (block) CFRelease(block);
    if (status != noErr) {
        set_error(error, nil, [NSString stringWithFormat:@"Could not create Apple audio sample (OSStatus %d)", (int)status]);
    }
    return sample;
}

bool upmixer_audio_schedule(UpmixerAudioHost opaque, const float *const *channels,
                            uint32_t channel_count, uint32_t frames, char **error) {
    @autoreleasepool {
        UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
        if (!host || !channels || channel_count != host.format.channelCount || !frames || frames > BUFFER_FRAMES) {
            set_error(error, nil, @"Native audio buffer format mismatch");
            return false;
        }
        for (uint32_t channel = 0; channel < channel_count; channel++) {
            if (!channels[channel]) {
                set_error(error, nil, @"Missing native audio channel");
                return false;
            }
        }
        if (host.renderer) {
            __block BOOL ok = NO;
            dispatch_sync(host.queue, ^{
                if (!media_ok(host, error)) return;
                if (host.nextFrame > INT64_MAX - frames
                    || host.pending.count >= MEDIA_QUEUE_FRAMES / BUFFER_FRAMES
                    || host.nextFrame - media_frame(host) + frames > MEDIA_QUEUE_FRAMES) {
                    set_error(error, nil, @"Apple audio queue is full");
                    return;
                }
                if (host.started && host.playing && media_frame(host) > host.nextFrame + BUFFER_FRAMES) {
                    set_error(error, nil, @"Apple audio underrun: realtime rendering could not keep up");
                    return;
                }
                CMSampleBufferRef sample = media_sample(host, channels, frames, error);
                if (!sample) return;
                [host.pending addObject:CFBridgingRelease(sample)];
                host.finished = NO;
                host.nextFrame += frames;
                request_media(host);
                start_media_if_ready(host);
                ok = YES;
            });
            return ok;
        }
        if (dispatch_semaphore_wait(host.semaphore, dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC))) {
            set_error(error, nil, @"Direct audio output stopped accepting buffers");
            return false;
        }
        AVAudioPCMBuffer *buffer;
        @synchronized(host.available) {
            buffer = host.available.lastObject;
            [host.available removeLastObject];
        }
        buffer.frameLength = frames;
        for (uint32_t channel = 0; channel < channel_count; channel++) {
            memcpy(buffer.floatChannelData[channel], channels[source_channel(host, channel)], frames * sizeof(float));
        }
        [host.player scheduleBuffer:buffer completionCallbackType:AVAudioPlayerNodeCompletionDataRendered
            completionHandler:^(AVAudioPlayerNodeCompletionCallbackType condition) {
                (void)condition;
                @synchronized(host.available) { [host.available addObject:buffer]; }
                dispatch_semaphore_signal(host.semaphore);
            }];
        return true;
    }
}

int64_t upmixer_audio_playback_frame(UpmixerAudioHost opaque) {
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (!host.renderer) return -1;
    __block int64_t frame;
    dispatch_sync(host.queue, ^{ frame = MIN(host.nextFrame, media_frame(host)); });
    return frame;
}

int upmixer_audio_finish(UpmixerAudioHost opaque, char **error) {
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (!host.renderer) return 1;
    __block int drained;
    dispatch_sync(host.queue, ^{
        host.finished = YES;
        start_media_if_ready(host);
        drained = media_ok(host, error) ? (host.pending.count == 0 && media_frame(host) >= host.nextFrame) : -1;
    });
    return drained;
}

void upmixer_audio_destroy(UpmixerAudioHost opaque) {
    if (!opaque) return;
    UpmixerAudio *host = (__bridge_transfer UpmixerAudio *)opaque;
    if (host.renderer) {
        [[NSNotificationCenter defaultCenter] removeObserver:host.flushObserver];
        dispatch_sync(host.queue, ^{
            host.playing = NO;
            [host.renderer stopRequestingMediaData];
            host.synchronizer.rate = 0.0f;
            [host.renderer flush];
            [host.pending removeAllObjects];
        });
    } else {
        [host.player stop];
        [host.engine stop];
    }
}

void upmixer_audio_free_error(char *error) { free(error); }

uint32_t upmixer_audio_max_output_channels(void) {
    @autoreleasepool {
        AVAudioEngine *engine = [AVAudioEngine new];
        return [[engine.outputNode outputFormatForBus:0] channelCount];
    }
}
