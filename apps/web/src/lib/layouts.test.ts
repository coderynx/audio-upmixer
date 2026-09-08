import { describe, expect, it } from "vitest";
import { ADM_DELIVERY_LAYOUTS, deliveryTypeForLayout, isStereoLayout, outputModeForLayoutSwitch } from "./layouts";

describe("isStereoLayout", () => {
  it("is true only for the literal stereo layout", () => {
    expect(isStereoLayout("stereo")).toBe(true);
    expect(isStereoLayout("5.1")).toBe(false);
    expect(isStereoLayout(undefined)).toBe(false);
  });
});

describe("deliveryTypeForLayout", () => {
  it("retargets a bed-collapsing delivery type back to multichannel for a stereo layout", () => {
    expect(deliveryTypeForLayout("stereo", "binaural")).toBe("multichannel");
  });

  it("leaves the delivery type alone for a multichannel layout", () => {
    expect(deliveryTypeForLayout("5.1", "binaural")).toBe("binaural");
  });

  it("accepts ADM delivery for every supported surround layout", () => {
    expect(ADM_DELIVERY_LAYOUTS).toEqual(["5.1", "7.1", "5.1.2", "5.1.4", "7.1.2", "7.1.4"]);
    for (const layout of ADM_DELIVERY_LAYOUTS) {
      expect(deliveryTypeForLayout(layout, "adm-bwf")).toBe("adm-bwf");
    }
  });

  it("retargets ADM for stereo and unknown layouts", () => {
    expect(deliveryTypeForLayout("stereo", "adm-bwf")).toBe("multichannel");
    expect(deliveryTypeForLayout("9.1.6", "adm-bwf")).toBe("multichannel");
  });
});

describe("outputModeForLayoutSwitch", () => {
  it("picks native when the device supports the switched-to layout's channel count", () => {
    expect(outputModeForLayoutSwitch(true)).toEqual({ outputMode: "native" });
  });

  it("falls back to binaural at the flat profile when the device does not", () => {
    expect(outputModeForLayoutSwitch(false)).toEqual({ outputMode: "binaural", spatialProfile: "flat" });
  });
});
