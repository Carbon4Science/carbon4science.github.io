# Carbon4Science project handoff

Last updated: 2026-07-23 (Asia/Seoul)

이 문서는 새 세션에서 프로젝트 작업을 이어가기 위한 현재 맥락이다.

## 현재 benchmark 방향

- finetuning 및 finetuned model benchmark는 현재 범위에서 제외한다.
- pretrained 8개 모델을 benchmark한다.
  - eSEN-30M-MP (`eSEN`)
  - ORB v2 MPtrj (`ORB`)
  - DPA-4.0-Pro-MPtrj (`DPA4`)
  - NequIP-MP-L (`NequIP`)
  - MACE-MP-0 (`MACE`)
  - SevenNet-l3i5 (`SevenNet`)
  - Nequix MP (`Nequix`)
  - CHGNet (`CHGNet`)
- 기존 한 물질 LGPS RDF/MSD benchmark는 삭제하지 않았다. 새 dynamat benchmark와 분리했고, legacy 스크립트로 보존했다.
- metric은 CPS가 아니라 CMDS이다. CMDS 값은 `MLIP/dynamat_metrics.py`의 `MATBENCH_DISCOVERY_METRICS` dictionary에 직접 입력한다. 현재 값은 `None` placeholder다.
- carbon cost만 직접 측정하며, 구조별 3 seed 평균 후 17개 구조 평균을 모델 최종값으로 사용한다.
- 기본 MD 설정은 timestep 2 fs, equilibration 10 ps, production 50 ps, seeds 42/43/44이다.

## 구조 입력

- 원본 HDF5:
  `MLIP/dynamat_trajectory/md_2026-06-29-dynamat-v1.0-reference-trajectories.h5`
- Matbench Discovery dynamat 구조 17개를 초기 프레임 CIF로 미리 추출했다.
- CIF 위치:
  `MLIP/dynamat_initial_structures/*.cif`
- 추출 스크립트:
  `MLIP/prepare_dynamat_structures.py`
- benchmark는 CIF가 있으면 HDF5 대신 CIF를 읽는다. 따라서 일반 model environment에는 h5py가 필요 없다. 원본 HDF5는 보존한다.
- 온도는 CIF 내용이 아니라 파일명에 포함된 `_숫자K_` 표기에서 추출한다. 예: `anthracene_293K_Sharma_S.cif` → 293 K. 파일명에서 온도 표기를 제거/변경하지 않는다.

## 주요 코드

- 새 benchmark: `MLIP/dynamat_benchmark.py`
- CMDS placeholder: `MLIP/dynamat_metrics.py`
- Slurm 제출용 wrapper: `MLIP/benchmarks/slurm_benchmark.sh`
- dynamat 실행 helper: `MLIP/benchmarks/slurm_dynamat_benchmark.sh`
- 터미널 dry-run helper: `MLIP/benchmarks/run_dynamat.sh`
- 새 benchmark 설명: `MLIP/benchmarks/DYNAMAT.md`
- DPA4 inference: `MLIP/DPA4/Inference.py`
- DPA4 환경 설치 스크립트: `MLIP/DPA4/setup_env.sh`

## 출력 구조

benchmark 실행 후 다음 구조가 생성된다.

```text
MLIP/dynamat_results/
  <model>/
    <structure>/
      seed-42.traj
      seed-43.traj
      seed-44.traj
      result.json
    summary.json
```

`result.json`에는 seed별 runtime/carbon 결과가, `summary.json`에는 구조별 결과와 구조 평균 carbon cost, CMDS, MD 설정이 저장된다.

## DPA4 환경 상태

- conda environment: `dpa4`
- PyTorch: `2.11.0+cu128`
- DeepMD-kit: git tag `v3.2.0b0`
- `e3nn`: installed
- CUDA module: `cuda/12.8.1`
- 체크포인트:
  `MLIP/DPA4/dpa-4.0-pro-mptrj-21.88-32.10.pt`
- DeepMD-kit은 `DP_ENABLE_PYTORCH=1`로 CUDA 지원 빌드했다.
- 확인 결과 `enable_pytorch=1`, PyTorch custom operator 등록, DPA4 calculator 객체 생성이 모두 성공했다.
- `dp --pt show <checkpoint> model-branch`는 singletask 모델에 잘못된 옵션이므로 실패한다. 이는 checkpoint 오류가 아니다.
- 직접 DPA4를 실행할 때는 보통 다음이 필요하다.

```bash
module load cuda/12.8.1
conda activate dpa4
```

Slurm dynamat helper에는 CUDA module load가 이미 들어 있다.

## 실행 방법

최종 benchmark는 아직 제출하지 않았다. 실행 시에는 `slurm_dynamat_benchmark.sh`를 직접 제출하지 말고 Slurm wrapper를 제출한다.

```bash
sbatch MLIP/benchmarks/slurm_benchmark.sh all
```

기존 invocation 호환을 위해 아래도 `pretrained`를 무시하고 동작한다.

```bash
sbatch MLIP/benchmarks/slurm_benchmark.sh all pretrained
```

단일 모델 예:

```bash
sbatch MLIP/benchmarks/slurm_benchmark.sh DPA4
```

MD 시간을 조절하는 별도 config 파일은 아직 없고 CLI 인자를 사용한다.

```bash
sbatch MLIP/benchmarks/slurm_benchmark.sh DPA4 \
  --equilibration-ps 0 \
  --production-ps 10 \
  --timestep-fs 2 \
  --seeds 42 43 44
```

`equilibration-ps 0`이면 equilibration을 건너뛰고 production만 실행하며, CarbonTracker도 production 구간에 대해서만 시작/종료된다.

## Slurm 전 사전 검증

실제 MD 없이 calculator와 CIF 구조만 검증하려면 터미널에서 실행한다.

```bash
bash MLIP/benchmarks/run_dynamat.sh CHGNet --dry-run
bash MLIP/benchmarks/run_dynamat.sh DPA4 --dry-run
```

구조 하나만 빠르게 확인할 수도 있다.

```bash
bash MLIP/benchmarks/run_dynamat.sh CHGNet --dry-run \
  --structures CsSnI3_500K_Ivor_VASP
```

`--dry-run`은 model calculator 로드, 17개 CIF 읽기, 온도 파일명 검증만 수행하며 MD, CarbonTracker, trajectory 쓰기는 하지 않는다. 8개 모델 모두를 확인하려면 `bash MLIP/benchmarks/slurm_dynamat_benchmark.sh all --dry-run`을 사용할 수 있지만, 각 모델의 pretrained weight 로딩/다운로드가 발생할 수 있다.

## 실행 전 주의사항

- GPU/CPU를 많이 사용하는 MD benchmark는 사용자 확인 없이 실행하거나 제출하지 않는다.
- 최종 `all` 실행은 8개 모델 × 17개 구조 × 3 seed를 순차적으로 수행하므로 시간이 오래 걸릴 수 있다.
- 실행 전 각 model environment의 calculator import와 checkpoint 경로를 가볍게 점검하는 것이 좋다.
- CMDS 값 입력 전에는 결과의 metric field가 `None`으로 남는다.
- `slurm_benchmark.sh` 기본 경로는 새 dynamat benchmark이며, 기존 RDF/MSD 작업은 명시적으로 `DYNAMAT_LEGACY=1`을 사용할 때만 실행한다.
