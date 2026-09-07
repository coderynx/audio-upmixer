// Run with apps/web/scripts/test-native-audio.sh; optional --capture uses BlackHole 2ch.
#import "../audio_bridge.m"
#import <AudioToolbox/AudioToolbox.h>
#import <assert.h>
#import <math.h>
#import <stdatomic.h>
#import <unistd.h>

static void check(BOOL ok, char **error) {
    if (!ok) fprintf(stderr, "FAIL: %s\n", *error ?: "assertion failed");
    assert(ok);
    assert(!*error);
}

static void test_samples(void) {
    for (NSString *layout in @[@"5.1", @"7.1", @"5.1.2", @"5.1.4", @"7.1.2", @"7.1.4"]) {
        char *error = NULL;
        UpmixerAudioHost opaque = upmixer_audio_create(layout.UTF8String, true, true, 96000, &error);
        check(opaque != NULL, &error);
        UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
        float data[12][127];
        const float *channels[12];
        for (int c = 0; c < 12; c++) {
            channels[c] = data[c];
            for (int f = 0; f < 127; f++) data[c][f] = c + f / 256.0f;
        }
        CMSampleBufferRef sample = media_sample(host, channels, 127, &error);
        check(sample != NULL, &error);
        assert(CMSampleBufferGetNumSamples(sample) == 127);
        assert(CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), CMTimeMake(2, 1)) == 0);
        assert(CMTimeCompare(CMSampleBufferGetDuration(sample), CMTimeMake(127, 48000)) == 0);
        const AudioStreamBasicDescription *asbd = CMAudioFormatDescriptionGetStreamBasicDescription(CMSampleBufferGetFormatDescription(sample));
        assert(asbd->mSampleRate == 48000 && asbd->mBytesPerFrame == host.format.channelCount * 4);
        assert(!(asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved));
        char *bytes = NULL;
        assert(CMBlockBufferGetDataPointer(CMSampleBufferGetDataBuffer(sample), 0, NULL, NULL, &bytes) == noErr);
        int map714[] = {0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11};
        for (unsigned c = 0; c < host.format.channelCount; c++) {
            int source = [layout isEqualToString:@"7.1.4"] ? map714[c] : c;
            for (int f = 0; f < 127; f++) assert(((float *)bytes)[f * host.format.channelCount + c] == data[source][f]);
        }
        CFRelease(sample);
        upmixer_audio_destroy(opaque);
    }
    char *error = NULL;
    assert(!upmixer_audio_create("invalid", true, true, 0, &error));
    assert(error);
    free(error);
    puts("PASS: layout order, LFE preservation, interleaving, timestamps and partial buffers");
}

static void test_queue_bounds(void) {
    char *error = NULL;
    UpmixerAudioHost opaque = upmixer_audio_create("7.1.4", true, false, 0, &error);
    check(opaque != NULL, &error);
    float silence[512] = {0};
    const float *channels[12];
    for (int c = 0; c < 12; c++) channels[c] = silence;
    for (int i = 0; i < MEDIA_QUEUE_FRAMES / 512; i++) {
        assert(upmixer_audio_ready(opaque, &error) == 1);
        check(upmixer_audio_schedule(opaque, channels, 12, 512, &error), &error);
    }
    assert(upmixer_audio_playback_frame(opaque) == 0);
    assert(upmixer_audio_ready(opaque, &error) == 0);
    assert(!upmixer_audio_schedule(opaque, channels, 12, 512, &error));
    assert(error);
    free(error);
    error = NULL;
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    [[NSNotificationCenter defaultCenter] postNotificationName:AVSampleBufferAudioRendererWasFlushedAutomaticallyNotification
                                                        object:host.renderer];
    assert(upmixer_audio_ready(opaque, &error) == -1);
    assert(error && strstr(error, "reset"));
    free(error);
    upmixer_audio_destroy(opaque);
    error = NULL;
    opaque = upmixer_audio_create("7.1.4", true, false, 0, &error);
    check(opaque != NULL, &error);
    check(upmixer_audio_schedule(opaque, channels, 12, 127, &error), &error);
    assert(upmixer_audio_finish(opaque, &error) == 0);
    check(upmixer_audio_schedule(opaque, channels, 12, 127, &error), &error);
    upmixer_audio_destroy(opaque);
    puts("PASS: bounded paused queue, loop continuation after EOF, and explicit output-reset error");
}

static float captured[48000 * 12 * 2];
static atomic_uint capturedCount;
static void capture(void *context, AudioQueueRef queue, AudioQueueBufferRef buffer,
                    const AudioTimeStamp *time, UInt32 packets, const AudioStreamPacketDescription *desc) {
    (void)context; (void)time; (void)packets; (void)desc;
    unsigned count = buffer->mAudioDataByteSize / sizeof(float);
    unsigned offset = atomic_load(&capturedCount);
    if (offset + count <= sizeof(captured) / sizeof(float)) {
        memcpy(captured + offset, buffer->mAudioData, count * sizeof(float));
        atomic_store(&capturedCount, offset + count);
    }
    AudioQueueEnqueueBuffer(queue, buffer, 0, NULL);
}

static void reference_output(void *context, AudioQueueRef queue, AudioQueueBufferRef buffer) {
    (void)context;
    float *samples = buffer->mAudioData;
    for (int i = 0; i < 512; i++) samples[2*i] = samples[2*i+1] = 0.03 * sin(2*M_PI*440*i/48000);
    buffer->mAudioDataByteSize = 4096;
    AudioQueueEnqueueBuffer(queue, buffer, 0, NULL);
}

static NSString *blackhole_uid(void) {
    AudioObjectPropertyAddress property = {kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioDeviceID devices[64];
    UInt32 size = sizeof(devices);
    assert(AudioObjectGetPropertyData(kAudioObjectSystemObject, &property, 0, NULL, &size, devices) == noErr);
    for (unsigned i = 0; i < size / sizeof(AudioDeviceID); i++) {
        AudioObjectPropertyAddress nameProperty = {kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        CFStringRef name = NULL;
        UInt32 length = sizeof(name);
        if (AudioObjectGetPropertyData(devices[i], &nameProperty, 0, NULL, &length, &name) != noErr) continue;
        BOOL match = [(__bridge NSString *)name isEqualToString:@"BlackHole 2ch"];
        CFRelease(name);
        if (!match) continue;
        nameProperty.mSelector = kAudioDevicePropertyDeviceUID;
        CFStringRef uid = NULL;
        length = sizeof(uid);
        assert(AudioObjectGetPropertyData(devices[i], &nameProperty, 0, NULL, &length, &uid) == noErr);
        return CFBridgingRelease(uid);
    }
    return nil;
}

static void test_playback(NSString *device, int totalFrames, int64_t startFrame, BOOL pause) {
    char *error = NULL;
    UpmixerAudioHost opaque = upmixer_audio_create("7.1.4", true, true, startFrame, &error);
    check(opaque != NULL, &error);
    UpmixerAudio *host = (__bridge UpmixerAudio *)opaque;
    if (device) host.renderer.audioOutputDeviceUniqueID = device;
    check(upmixer_audio_start(opaque, &error), &error);
    upmixer_audio_resume(opaque);
    float data[12][512] = {0};
    const float *channels[12];
    for (int c = 0; c < 12; c++) channels[c] = data[c];
    int offset = 0;
    BOOL paused = NO;
    double began = monotonic_seconds();
    double started = 0;
    int64_t previous = startFrame;
    while (offset < totalFrames) {
        assert(monotonic_seconds() - began < 10);
        int ready = upmixer_audio_ready(opaque, &error);
        check(ready >= 0, &error);
        int64_t position = upmixer_audio_playback_frame(opaque);
        assert(position >= previous && position <= startFrame + offset);
        previous = position;
        if (pause && !paused && position > startFrame + 48000) {
            upmixer_audio_pause(opaque);
            usleep(30000);
            int64_t stopped = upmixer_audio_playback_frame(opaque);
            usleep(150000);
            assert(llabs(upmixer_audio_playback_frame(opaque) - stopped) < 64);
            upmixer_audio_resume(opaque);
            paused = YES;
        }
        if (!ready) { usleep(1000); continue; }
        unsigned frames = MIN(512, totalFrames - offset);
        for (unsigned f = 0; f < frames; f++) data[0][f] = 0.03f * sin(2 * M_PI * 440 * (offset + f) / 48000.0);
        check(upmixer_audio_schedule(opaque, channels, 12, frames, &error), &error);
        offset += frames;
        if (offset < MEDIA_PREFILL_FRAMES) assert(upmixer_audio_playback_frame(opaque) == startFrame);
        if (!started && offset >= MEDIA_PREFILL_FRAMES) started = monotonic_seconds();
        // Simulate DSP work and bounded scheduler jitter, still faster than realtime.
        usleep(offset % 8192 == 0 ? 15000 : 1000);
    }
    if (!started) started = monotonic_seconds();
    while (true) {
        int done = upmixer_audio_finish(opaque, &error);
        check(done >= 0, &error);
        if (done) break;
        assert(monotonic_seconds() - began < 10);
        usleep(1000);
    }
    double elapsed = monotonic_seconds() - started - (paused ? 0.18 : 0);
    assert(fabs(elapsed - totalFrames / 48000.0) < 0.12);
    assert(upmixer_audio_playback_frame(opaque) == startFrame + totalFrames);
    upmixer_audio_pause(opaque);
    upmixer_audio_destroy(opaque);
    printf("PASS: %d frames, start=%lld, pause=%d, elapsed=%.3fs (expected %.3fs)\n",
           totalFrames, startFrame, pause, elapsed, totalFrames / 48000.0);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        test_samples();
        test_queue_bounds();
        if (argc == 1) return 0;
        assert(argc == 2 && (strcmp(argv[1], "--capture") == 0 || strcmp(argv[1], "--playback") == 0));
        NSString *device = blackhole_uid();
        assert(device && "Install BlackHole 2ch for the output capture check");
        if (strcmp(argv[1], "--playback") == 0) {
            test_playback(device, 144137, 0, NO);
            test_playback(device, 144137, 96000, YES);
            test_playback(device, 127, 480000, NO);
            return 0;
        }
        __block BOOL permissionDone = NO;
        __block BOOL permissionGranted = NO;
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                permissionGranted = granted;
                permissionDone = YES;
            });
        }];
        while (!permissionDone) {
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
        }
        if (!permissionGranted) {
            fprintf(stderr, "BLOCKED: allow Upmixer Audio Test in System Settings > Privacy & Security > Microphone\n");
            return 77;
        }
        AudioStreamBasicDescription format = {48000, kAudioFormatLinearPCM,
            kAudioFormatFlagsNativeFloatPacked, 8, 1, 8, 2, 32, 0};
        AudioQueueRef queue;
        assert(AudioQueueNewInput(&format, capture, NULL, NULL, NULL, 0, &queue) == noErr);
        CFStringRef uid = (__bridge CFStringRef)device;
        assert(AudioQueueSetProperty(queue, kAudioQueueProperty_CurrentDevice, &uid, sizeof(uid)) == noErr);
        AudioQueueBufferRef captureBuffers[3];
        for (int i = 0; i < 3; i++) {
            assert(AudioQueueAllocateBuffer(queue, 4096, &captureBuffers[i]) == noErr);
            assert(AudioQueueEnqueueBuffer(queue, captureBuffers[i], 0, NULL) == noErr);
        }
        assert(AudioQueueStart(queue, NULL) == noErr);
        usleep(100000);
        AudioQueueRef reference;
        assert(AudioQueueNewOutput(&format, reference_output, NULL, NULL, NULL, 0, &reference) == noErr);
        assert(AudioQueueSetProperty(reference, kAudioQueueProperty_CurrentDevice, &uid, sizeof(uid)) == noErr);
        for (int i = 0; i < 3; i++) {
            AudioQueueBufferRef buffer;
            assert(AudioQueueAllocateBuffer(reference, 4096, &buffer) == noErr);
            reference_output(NULL, reference, buffer);
        }
        assert(AudioQueueStart(reference, NULL) == noErr);
        usleep(300000);
        AudioQueueStop(reference, true);
        AudioQueueDispose(reference, true);
        usleep(100000);
        AudioQueueStop(queue, true);
        double referencePower = 0;
        unsigned referenceCount = atomic_load(&capturedCount);
        for (unsigned i = 0; i < referenceCount; i++) referencePower += captured[i] * captured[i];
        fprintf(stderr, "reference samples=%u power=%g\n", referenceCount, referencePower);
        assert(referenceCount > 0 && referencePower / referenceCount > 1e-12);
        atomic_store(&capturedCount, 0);
        for (int i = 0; i < 3; i++) assert(AudioQueueEnqueueBuffer(queue, captureBuffers[i], 0, NULL) == noErr);
        assert(AudioQueueStart(queue, NULL) == noErr);
        test_playback(device, 144137, 0, NO);
        usleep(250000);
        assert(AudioQueueStop(queue, true) == noErr);
        assert(AudioQueueDispose(queue, true) == noErr);
        unsigned frames = atomic_load(&capturedCount) / 2;
        unsigned active = 0, first = 0, last = 0, crossings = 0;
        for (unsigned i = 480; i < frames; i += 480) {
            double power = 0;
            for (unsigned j = i - 480; j < i; j++) power += captured[j * 2] * captured[j * 2];
            if (power / 480 > referencePower / referenceCount * 0.05) {
                if (!active) first = i - 480;
                last = i;
                active++;
            }
        }
        fprintf(stderr, "capture frames=%u active=%u first=%u last=%u\n", frames, active, first, last);
        assert(active > 290 && active < 310);
        assert((last - first) / 480 - active <= 1);
        for (unsigned i = first + 1; i < last; i++) {
            if (captured[(i - 1) * 2] <= 0 && captured[i * 2] > 0) crossings++;
        }
        double frequency = crossings * 48000.0 / (last - first);
        assert(fabs(frequency - 440) < 3);
        printf("PASS: captured %.2fs, %.2fHz, no internal 10ms gaps\n", active / 100.0, frequency);
        test_playback(device, 144137, 96000, YES);
        test_playback(device, 127, 480000, NO);
        puts("PASS: capture and transport checks complete");
    }
}
