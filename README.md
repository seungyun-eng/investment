# Stock Research Project

기존 Jupyter Notebook들을 **하나의 GitHub 저장소 안에서 기능별로 통합**한 프로젝트입니다.

중요한 점은 **모든 기능이 한 파일에 들어 있지는 않으며**, 원하는 작업만 따로 실행합니다.

## 권장 OneDrive 구조

```text
OneDrive/주식/
├── stock-project/              # 이 Git 저장소
├── Back Test/                  # 원본 주가
├── Processed Data/             # 지표 계산 결과
├── Macro Data/                 # VIX, CPI 등
├── Financial_Data_real/        # 재무제표와 분석
└── Results/
    ├── Parameters/
    └── Transformer/
```

저장소가 `OneDrive/주식/stock-project`에 있으면 경로를 자동으로 찾습니다.  
다른 구조를 사용하면 `.env` 또는 환경변수 `STOCK_DATA_ROOT`를 지정합니다.

## 처음 한 번 설정

PowerShell에서 저장소 폴더로 이동한 뒤:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
pytest
```

Transformer까지 사용할 컴퓨터에서는:

```powershell
pip install -e .[ml,dev]
```

`.venv`는 컴퓨터마다 새로 만들고 GitHub에는 올리지 않습니다.

## 각 기능을 따로 실행

### 1. Macro 데이터 업데이트

```powershell
python scripts/update_macro.py
```

### 2. 주가 업데이트

전체:

```powershell
python scripts/update_prices.py
```

일부 종목:

```powershell
python scripts/update_prices.py --ticker TSLA --ticker NVDA
```

### 3. 주가 전처리 및 지표 계산

```powershell
python scripts/preprocess_prices.py
```

### 4. 재무제표 크롤링

```powershell
python scripts/scrape_financials.py --ticker TSLA
```

Chrome/Selenium이 필요하며 사이트 구조가 바뀌면 수정이 필요할 수 있습니다.

### 5. 재무 분석과 DCF

```powershell
python scripts/analyze_financials.py --ticker TSLA
```

### 6. 재무 그래프

```powershell
python scripts/visualize_financials.py TSLA
```

### 7. VIX 전략 최적화

`company`는 `Processed Data` 파일명 앞부분과 일치해야 합니다.

```powershell
python scripts/optimize_vix.py Tesla --start 2022-01-01 --end 2025-06-24
```

빠른 테스트:

```powershell
python scripts/optimize_vix.py Tesla --start 2022-01-01 --end 2025-06-24 --tpe-trials 20 --cma-trials 10
```

결과는 `Results/Parameters/vix_parameters.xlsx`에 저장됩니다.

### 8. VIX 전략 시뮬레이션

```powershell
python scripts/simulate_vix.py 1 --start 2020-01-01 --end 2025-12-31
```

### 9. Technical 전략 최적화

```powershell
python scripts/optimize_technical.py Tesla --start 2022-01-01 --end 2025-06-24
```

결과는 `Results/Parameters/technical_parameters.xlsx`에 저장됩니다.

### 10. Technical 전략 시뮬레이션

```powershell
python scripts/simulate_technical.py 1 --start 2020-01-01 --end 2025-12-31
```

### 11. 구형 Grid Search 실험

```powershell
python scripts/run_legacy_grid_search.py Tesla
```

이 실행은 `Back Test Final` 계열의 과거 실험을 보존한 별도 경로입니다.

### 12. Transformer

```powershell
python scripts/train_transformer.py Tesla --epochs 50 --horizon 20
```

## GitHub 운영

```powershell
git status
git add .
git commit -m "refactor: integrate stock notebooks"
git push
```

다른 컴퓨터에서는 OneDrive 동기화가 끝난 뒤 작업합니다. 같은 저장소를 두 컴퓨터에서 동시에 수정하지 마세요.

## 원본 Notebook

`archive/original_notebooks`에는 출력 결과를 제거한 원본 Notebook이 있습니다.  
코드 기준은 `docs/NOTEBOOK_MAP.md`에 기록했습니다.

## 중요한 검증 사항

이 프로젝트는 업로드된 Notebook의 주요 기능을 모듈화한 **통합 v1**입니다.  
실제 OneDrive 데이터의 컬럼과 파일명을 사용한 전체 실행 검증은 아직 필요합니다.

먼저 다음 순서로 확인하세요.

1. `pytest`
2. `python scripts/preprocess_prices.py`
3. 한 종목으로 20회 수준의 작은 Optuna 테스트
4. 기존 Notebook 결과와 ROI 및 거래일 비교
5. 일치 확인 후 정식 trial 수로 실행
