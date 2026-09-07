# API

The API context persists web work and delivers it through background processing.

## Language

**Project**: A persisted editable collection of tracks and layout-specific mixes.
_Avoid_: job

**Track**: One source within a project with its own mix state.
_Avoid_: stem

**Job**: An asynchronous render request with owned inputs and outputs.
_Avoid_: project

**Prepared stems**: Project-owned stem assets available for editing and export.
_Avoid_: shared cache
