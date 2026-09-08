import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { StemDynamicEqWindow } from "./StemDynamicEqWindow";

it("builds editor controls only while open, using the latest values", () => {
  let frequency = 1000;
  const readFrequency = vi.fn(() => frequency);
  const value = {
    enabled: true, profile: null, mix: 100,
    bands: [{
      enabled: true, get freq_hz() { return readFrequency(); },
      q: 1, threshold_db: -24, ratio: 2, max_cut_db: 3, attack_ms: 20, release_ms: 200,
    }],
  };
  const onChange = vi.fn();
  const editor = () => <StemDynamicEqWindow stemName="Vocals" value={value} onChange={onChange} />;
  const { rerender } = render(editor());
  frequency = 2000;
  rerender(editor());
  expect(readFrequency).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Open tame dynamic EQ" }));
  expect(readFrequency).toHaveBeenCalled();
  expect(screen.getByText("2000 Hz")).toBeInTheDocument();
  fireEvent.keyDown(screen.getByRole("slider", { name: "Ratio 1" }), { key: "ArrowRight" });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
    bands: [expect.objectContaining({ ratio: 2.1, freq_hz: 2000 }), expect.any(Object)],
  }));
  fireEvent.click(screen.getByRole("button", { name: "Close Tame" }));
  readFrequency.mockClear();
  rerender(editor());
  expect(readFrequency).not.toHaveBeenCalled();
});
