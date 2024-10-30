import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

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

# 도로 이미지와 마스크 이미지 불러오기
image = cv2.imread('content/test/images/val_image (262).png')
image_height, image_width = image.shape[:2]

# 이진 마스크 불러오기 (도로와 장애물 구분)
binary_mask = cv2.imread('content/test/labels/val_labels (262).png', cv2.IMREAD_GRAYSCALE)
binary_mask = (binary_mask == 0).astype(np.uint8)  # 도로 부분은 0, 장애물 부분은 1

# 1. 마스크를 컬러로 변환
mask_color = np.zeros_like(image)  # 마스크와 같은 크기의 빈 배열 생성
mask_color[binary_mask == 0] = [0, 255, 0]  # 도로 부분을 초록색으로 설정

# 2. 원본 이미지와 마스크 이미지를 오버레이
overlay_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)  # 가중치를 조정하여 오버레이

# 시작점과 목표점 정의
start = (786, 1200)  # 시작점 (y, x)

x_start = start[1]
'''
# goal : x에 가까우면서 y가 가장 먼 지점 (y는 min을 써야함)
road_coords = [(y, x) for y, x in zip(*np.where(binary_mask == 0)) if x == x_start]
if road_coords:
    goal = min(road_coords, key=lambda coord: coord[0])  # y 값이 가장 큰 도로 좌표 선택
else:
    print("No valid goal found at the given x position.")
    goal = start  # 경로가 없을 경우 시작점으로 goal 설정
'''

# start의 x 좌표 설정
x_start = start[1]

# goal 설정: y가 450 이상이고 x가 start와 비슷한 가장 큰 도로 좌표 찾기
road_coords = [(y, x) for y in range(420, binary_mask.shape[0]) 
               for x in range(binary_mask.shape[1]) 
               if binary_mask[y, x] == 0 and abs(x - x_start) <= 300]

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
path = astar(start, goal, binary_mask)

# 경로 시각화 (굵게 표시)
if path:
    for i in range(len(path) - 1):
        cv2.arrowedLine(overlay_image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), 
                 (0, 0, 255), thickness=10)  # 경로를 빨간색으로 설정하고 두께를 3으로 설정
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
plt.title("Overlay of Mask and A* Path on Original Image")
plt.axis("off")
plt.show()
