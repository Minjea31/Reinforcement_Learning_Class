# 발표·설계 보고서 제작 코드

과제 소스는 ../assignment/에 분리했다. 모든 수치·그림·영상은 새 OpenAI Gym 온라인 1-step Actor-Critic 실험의 결과를 읽는다.

1. python build.py
2. LibreOffice로 생성한 PPTX를 PDF로 변환
3. python build_report.py
4. python render_check.py
5. python check_presentation.py /home/a/ppt_workspace/output/RL_week11/RL_week11.pptx

build.py는 기본 탐색과 추가 탐색의 비교를 구분한다. build_report.py는 최신 발표 텍스트·그림과 실제 실험값으로 6쪽 PDF를 생성하고 실제 Gym 동영상을 첨부한다. render_check.py는 발표와 보고서 전 페이지 PNG 및 배치 검사 기록을 만든다. 사용자가 삭제한 내용은 복원하지 않는다. ZIP은 생성하지 않는다.

제작 작업 공간은 /home/a/ppt_workspace/ppt_info/RL_week11/, 최종 파일은 /home/a/ppt_workspace/output/RL_week11/이다.
