# 통합 과정에서 반영한 수정

## 경로

- `C:\Users\seung`, `C:\Users\LabPC`, `D:\주식` 하드코딩 제거
- 저장소가 들어 있는 OneDrive `주식` 폴더를 기준으로 상대 경로 계산
- 필요 시 `STOCK_DATA_ROOT` 환경변수로 덮어쓰기

## 실행 방식

- Notebook import 시 다운로드나 최적화가 자동 실행되던 구조 제거
- 각 작업을 `scripts/*.py`로 분리
- 전체 자동 실행 파일은 만들지 않음

## 전략과 백테스트

- VIX 최적화와 시뮬레이션이 서로 다른 매수 조건을 사용하던 문제 수정
- VIX 전략은 두 단계 모두 VIX + RSI + Bollinger + MACD 조건을 동일하게 사용
- 존재하지 않는 `tw_price_th`, `tm_price_th`, `tw_vol_th`, `tm_vol_th` 참조 제거
- Technical 시뮬레이션 결과 폴더가 생성되지 않던 문제 수정
- 매수 포지션이 이미 있는데도 같은 날/연속일에 다시 BUY 로그가 생기던 문제 수정
- ROI를 일관되게 순수익률 `(최종자산 / 총투입금 - 1) × 100`으로 계산
- VIX의 `vix_sell_th <= vix_buy_th` 제약을 TPE와 CMA 모두에 적용
- VIX와 Technical 파라미터 Excel을 별도 파일로 분리

## 데이터

- 거래량 K/M/B 및 쉼표 숫자 처리
- `dropna()`로 200일 이전의 모든 데이터를 일괄 삭제하지 않고 전략별 필수 열만 사용
- 주가 업데이트 시 종목 폴더의 모든 CSV를 삭제하던 위험한 로직 제거
- 이 프로젝트가 관리하는 Historical Data 파일만 교체
- OneDrive 동기화 중 부분 파일 노출을 줄이도록 임시 파일 후 `os.replace` 방식 사용

## Baseline

- Buy and Hold를 실제 기준선으로 추가
- 최저점 매수/이후 최고점 매도는 look-ahead가 있으므로 `PERFECT_FORESIGHT_DIAGNOSTIC`로 명확히 표시
- DCA는 실제 거래일마다 명시된 금액을 투자하도록 계산
