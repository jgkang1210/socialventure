# socialventure
 
## 2020 소셜벤처 경연대회

주제 : 물체 인식 기능을 이용한 시각장애인용 안경
기간 : 2020.07.13 ~ 2020.09.05

Opencv 와 아루코마커를 이용하여 시각장애인들이 사용할 수 있게 일체형 안경을 제작.
안경에는 라즈베리파이 zero w 를 탑재하여 아루코마커의 인식을 통해 주변 사물이 무엇인지 알려주고, 이를 앱을 통해 사용자에게 소리로 전달해준다.

## 영상
[![0](http://img.youtube.com/vi/N39_Hc4jSvc/0.jpg)](http://www.youtube.com/watch?v=N39_Hc4jSvc "socialventure")
https://youtu.be/N39_Hc4jSvc

## 기능

사용자가 어떤 물체를 잡았는지 탐지하는 기능.
--> 화면 속 모든 아루코 마커를 인식하여 속도 벡터를 검출해내고, 이 중 평균 속도보다 가장 빨리 움직이는 속도 벡터를 가지는 아루코 마커가 사용자가 잡은 물체임을 알려준다.

화면 속의 마커들의 상대위치를 통해 시각장애인이 찾고자 하는 물건이 어느 위치에 있는지 알려준다.
ex) 꽃병이 책의 오른쪽에 있으면 그 위치를 알려준다.
