# DPA-4.0.1-Pro-MPtrj

The new dynamat benchmark uses this directory for DPA4.  Supply the exact
checkpoint with `--dpa4-checkpoint PATH` or the `DPA4_MODEL` environment
variable.  The legacy `MLIP/DPA3/` integration is retained for old results.

Run `setup_env.sh` for the separate `dpa4` environment. It pins
DeePMD-kit to `v3.2.0b0`, which is the intended DPA4-compatible release for
this setup. The exact
`DPA-4.0-Pro-MPtrj` download URL is intentionally required by
`download_model.sh`; do not substitute a MatPES or OMat24 checkpoint.

DPA4 requires a recent PyTorch backend of DeePMD-kit.  The checkpoint format
must be supported by the installed `deepmd.calculator.DP` implementation.
