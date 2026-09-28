# MuJoCo 모델

`leg_and_arm.xml`은 `src/simulate.py`가 대상자 체형, 로봇 구조와 관절 범위에
맞춰 생성한 가장 최근 모델입니다. 시뮬레이션을 실행할 때 이 파일은 자동으로
덮어써집니다.

영구 설정은 XML을 직접 수정하지 말고 다음 위치에서 변경합니다.

- 공통 기본값: `src/config.py`
- XML 형상 생성: `src/build_xml.py`
- 실행별 설정: `src/simulate.py` 명령행 인자 또는 `src/rehab_ui.py`
