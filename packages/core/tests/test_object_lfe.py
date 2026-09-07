"""Objects stay full-range; only beds author an LFE send."""

import numpy as np
import pytest

from upmixer.config import UpmixConfig
from upmixer.formats import FORMAT_MAP
from upmixer.io.adm_writer import render_adm_programme
from upmixer.separation.stem_router import StemRouter


@pytest.mark.parametrize("output_type", ["multichannel", "adm-bwf"])
@pytest.mark.parametrize("mode", ["mono", "linked-stereo"])
@pytest.mark.parametrize("stem", ["Vocals", "Toms@front"])
def test_objects_ignore_legacy_lfe_sends(output_type, mode, stem):
    tone = 0.2 * np.sin(2 * np.pi * 60 * np.arange(48000) / 48000)
    audio = np.column_stack([tone, tone])
    programmes = []
    for send in (0.0, 1.0):
        config = UpmixConfig(
            output_format="5.1", output_type=output_type,
            stem_object_mode={stem: mode},
            stem_routing={stem: {"LFE": send}},
        )
        programmes.append(StemRouter(config, FORMAT_MAP["5.1"], 48000).route(
            {stem: audio}, len(audio)
        ))
    plain, legacy = programmes
    for channel in plain.bed:
        np.testing.assert_array_equal(plain.bed[channel], legacy.bed[channel])
    assert not np.any(legacy.bed["LFE"])
    if output_type == "adm-bwf":
        rendered = render_adm_programme(legacy.bed, FORMAT_MAP["5.1"], legacy.objects)
        assert not np.any(rendered["LFE"])
        assert max(np.max(np.abs(channel)) for channel in rendered.values()) > 0.1
        assert len(legacy.objects) == (1 if mode == "mono" else 2)
        for before, after in zip(plain.objects, legacy.objects):
            np.testing.assert_array_equal(before.audio, after.audio)
            assert np.max(np.abs(after.audio)) > 0.1
    else:
        assert max(np.max(np.abs(channel)) for channel in legacy.bed.values()) > 0.1
