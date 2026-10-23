# 오늘의 트레일

영어권 트레일러닝 팟캐스트·영상의 핵심 이야기를 한국어 카드로 읽는 앱 프로토타입입니다.
카드 한 장은 영어 원문 구절, 한국어 번역, 용어 풀이, 설명, 해석(별도 표기)으로 이어집니다.

## 파일

- `index.html` : 앱 화면 (오늘 / 지난 호 / 내 메모, 카드 읽기). 브라우저에서 바로 열립니다.
- `deck.html` : 카드 덱 단독 페이지.
- `src/app_template.html`, `src/build_app.py` : `deck.html`의 카드 데이터를 앱 틀에 넣어 `index.html`을 만드는 소스.

## 현재 호

Francesco Puppi, "Logic, Pain, and the Reality of Ultra Running".
카드 내용은 제공받은 오디오를 음성 인식으로 옮겨 읽고 정리한 것입니다. 영어 원문은 짧은 구절만 싣고, 번역과 설명은 AI가 옮겼습니다.
원본 오디오와 전사본은 저장소에 포함하지 않았습니다.

## 만드는 법

```
cd src
python3 build_app.py   # ../deck.html 의 카드 데이터로 ../index.html 을 다시 만듭니다
```
