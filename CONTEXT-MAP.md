# Context Map

## Contexts

- [Core](packages/core/CONTEXT.md): source processing, separation, routing, and delivery.
- [DSP](packages/dsp/CONTEXT.md): shared offline and realtime signal processing.
- [CLI](apps/cli/CONTEXT.md): command-line invocation and manifest runs.
- [API](apps/api/CONTEXT.md): persisted web work and asynchronous delivery.
- [Web](apps/web/CONTEXT.md): interactive project editing and preview.

## Relationships

- **Core ↔ DSP**: Core supplies processing policy; DSP provides shared signal stages.
- **CLI → Core**: CLI resolves user input into public core calls.
- **API → Core**: API persists and delivers work performed through public core calls.
- **Web → API and DSP**: Web controls API-backed work and previews through shared DSP.
