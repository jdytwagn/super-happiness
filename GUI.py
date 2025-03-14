import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# Mock user credentials (replace with actual authentication logic)
VALID_USERNAME = "ysfq"
VALID_PASSWORD = "12345"

# 加载训练好的 YOLOv8 模型
model = YOLO('runs/detect/train/weights/best.pt')

def login():
    global root, login_window

    # 获取输入的用户名和密码
    username = username_entry.get()
    password = password_entry.get()

    # 验证用户名和密码是否正确
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        # 关闭登录窗口
        login_window.destroy()

        # 创建主界面
        root = tk.Tk()
        root.title("YOLOv8 图片预测")

        # 设置界面大小
        root.geometry("700x600")  # 设置宽度为700像素，高度为600像素

        # 创建框架，用于放置选择的图片和预测的图片
        image_frame = tk.Frame(root)
        image_frame.pack(pady=20)

        # 显示选择的图片
        global img_label, result_label, result_img_label
        img_label = tk.Label(image_frame)
        img_label.pack(side=tk.LEFT, padx=20)

        # 显示预测结果的图片
        result_img_label = tk.Label(image_frame)
        result_img_label.pack(side=tk.RIGHT, padx=20)

        # 显示预测结果的类别
        result_label = tk.Label(root, text="")
        result_label.pack(pady=20)

        # 创建按钮，用于选择图片并进行预测
        predict_button = tk.Button(root, text="选择图片并预测", command=predict_image)
        predict_button.pack()

        # 运行主界面
        root.mainloop()
    else:
        messagebox.showerror("登录失败", "用户名或密码错误，请重新输入。")

def predict_image():
    # 打开文件对话框，选择要预测的图片
    file_path = filedialog.askopenfilename()

    if not file_path:
        return  # 如果未选择文件，则返回

    # 读取图片文件
    image = Image.open(file_path)
    image.thumbnail((320, 320))  # 调整图片大小以适应界面

    # 在界面上显示选定的图片
    img_label.image = ImageTk.PhotoImage(image)
    img_label.config(image=img_label.image)

    # 将图片转换成 OpenCV 格式
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 使用 YOLOv8 模型进行预测
    results = model.predict(source=img_cv2)

    # 检查是否有检测结果
    if len(results) > 0 and len(results[0].boxes) > 0:
        # 获取预测结果的类别
        class_names = model.names
        predicted_class_idx = results[0].boxes[0].cls.cpu().numpy()
        predicted_class_idx = int(np.squeeze(predicted_class_idx))  # 确保将其转换为标量值
        predicted_class = class_names[predicted_class_idx]
        # 在界面上显示预测的类别
        result_label.config(text=f"预测结果: {predicted_class}")
        # 在界面上显示预测的图片及其结果
        result_img = results[0].plot()  # 使用 YOLOv8 提供的方法绘制带有检测结果的图片
        result_image = Image.fromarray(result_img)
        result_image.thumbnail((320, 320))  # 缩放预测结果图片大小以适应界面
        result_image_tk = ImageTk.PhotoImage(result_image)

        result_img_label.config(image=result_image_tk)
        result_img_label.image = result_image_tk
    else:
        result_label.config(text="No objects detected")
        result_img_label.config(image='')
        result_img_label.image = None

# 创建登录界面
login_window = tk.Tk()
login_window.title("登录")

# 设置登录界面大小
login_window.geometry("700x600")

# 用户名标签和输入框
username_label = tk.Label(login_window, text="用户名:")
username_label.pack(pady=20)
username_entry = tk.Entry(login_window)
username_entry.pack(pady=10)

# 密码标签和输入框
password_label = tk.Label(login_window, text="密码:")
password_label.pack(pady=20)
password_entry = tk.Entry(login_window, show="*")
password_entry.pack(pady=10)

# 登录按钮
login_button = tk.Button(login_window, text="登录", command=login)
login_button.pack(pady=20)

# 运行登录界面
login_window.mainloop()
