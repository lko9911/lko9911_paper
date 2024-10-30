# 라이브러리 가져오기 
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# A* 알고리즘 관련 함수 정의
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan 거리

def astar(start, goal, grid):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))  # (f_score, point)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # 역순으로 반환

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 상, 하, 좌, 우
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:  # 장애물인 경우
                    continue

                tentative_g_score = g_score[current] + 1  # 인접 노드까지의 비용
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 경로 업데이트
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set.queue]:
                        open_set.put((f_score[neighbor], neighbor))

    return []  # 경로가 없는 경우

# YOLO 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "content/test/images/val_image (258).png"
yolo_model = YOLO("yolov10x.pt")

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)

# 도로 이미지와 마스크 이미지 불러오기
image = cv2.imread('content/test/images/val_image (258).png')
image_height, image_width = image.shape[:2]

# 이진 마스크 불러오기 (도로와 장애물 구분)
binary_mask = cv2.imread('content/test/labels/val_labels (258).png', cv2.IMREAD_GRAYSCALE)
binary_mask = (binary_mask == 0).astype(np.uint8)  # 도로 부분은 0, 장애물 부분은 1

# 1. 마스크를 컬러로 변환
mask_color = np.zeros_like(image)  # 마스크와 같은 크기의 빈 배열 생성
mask_color[binary_mask == 0] = [0, 255, 0]  # 도로 부분을 초록색으로 설정

# 2. 원본 이미지와 마스크 이미지를 오버레이
overlay_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)  # 가중치를 조정하여 오버레이

# 감지된 객체의 경계 상자를 오버레이 이미지에 추가 및 방해물 설정
obstacle_mask = np.copy(binary_mask)  # 방해물 마스크를 복사하여 업데이트
for result in results:
    boxes = result.boxes.xyxy  # 경계 상자 좌표 (x1, y1, x2, y2)
    class_ids = result.boxes.cls  # 클래스 ID
    names = result.names  # 클래스 이름

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)  # 좌표 정수 변환

        # 경계 상자 그리기 (다른 색상 사용)
        color = (255, 0, 0)  # 빨간색으로 설정
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)

        # 클래스 이름 추가
        label = f'{names[int(class_id)]}'  # 클래스 이름을 문자열로 변환
        cv2.putText(overlay_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 방해물 마스크 업데이트 (감지된 객체를 장애물로 추가)
        obstacle_mask[y1:y2, x1:x2] = 1  # 장애물 부분을 1로 설정

# 시작점과 목표점 정의
start = (786, 1200)  # 시작점 (y, x)

# goal 설정: y가 450 이상이고 x가 start와 비슷한 가장 큰 도로 좌표 찾기
road_coords = [(y, x) for y in range(500, binary_mask.shape[0]) 
               for x in range(binary_mask.shape[1]) 
               if obstacle_mask[y, x] == 0 and abs(x - start[1]) <= 350]

if road_coords:
    # y 값이 가장 작은 좌표들 중 x 값이 start에 가장 가까운 좌표 선택
     goal = min(road_coords, key=lambda coord: coord[0])
else:
    print("No valid road coordinates found.")
    goal = start  # 도로가 없을 경우 시작점으로 goal 설정

# 시작점과 목표점의 좌표 출력
print(f"Start Point: {start}")
print(f"Goal Point: {goal}")

# A* 알고리즘 실행
path = astar(start, goal, obstacle_mask)

# 경로 시각화 (굵게 표시)
if path:
    for i in range(len(path) - 1):
        cv2.arrowedLine(overlay_image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), 
                         (0, 0, 255), thickness=10)  # 경로를 빨간색으로 설정하고 두께를 10으로 설정
else:
    print("No path found!")

# 시작점과 목표점 수치적으로 표현 (좌표만 표시)
cv2.putText(overlay_image, f'Start: {start}', (start[1] + 10, start[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
cv2.putText(overlay_image, f'Goal: {goal}', (goal[1] + 10, goal[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.title("Overlay of Mask and A* Path on Original Image with YOLO Detections and Obstacles")
plt.axis("off")
plt.show()
