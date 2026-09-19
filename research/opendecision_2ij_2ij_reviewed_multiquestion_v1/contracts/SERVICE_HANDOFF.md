# Track S handoff (not Rust service completion)

The included `odij_serve.py` is a resident, bounded-line JSONL prediction worker for the new native study contract.
It loads one locked profile once. It is NOT the pinned Jev HTTP format and does not implement Rust transports or authentication.
The host must supply a bounded queue, tenant authorization, cancellation/admission, and its versioned compatibility adapter.
A deadline is checked before/after inference; an already-dispatched CUDA operation is not preempted.
Use the existing repository's exact wire types and fixtures before claiming S.2/S.3 complete. No public tunnel is opened.
