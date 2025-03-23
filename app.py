import io
import os
import warnings

import streamlit as st
from PIL import Image

from recomendation import get_img_recommend

# Lọc các cảnh báo liên quan đến use_column_width
warnings.filterwarnings("ignore", message=".*use_column_width.*")

st.title("Clothes Recommendation System")

uploaded_file = st.file_uploader("Chọn ảnh của bạn", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh đầu vào (nếu bạn vẫn muốn hiển thị)
    if "loaded_images_bytes" in st.session_state:
        del st.session_state.loaded_images_bytes
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Ảnh đầu vào", use_container_width=True)

    st.markdown("---")
    st.subheader("Ảnh gợi ý")

    # Lấy danh sách đường dẫn ảnh gợi ý
    label, img_paths = get_img_recommend(uploaded_file)

    if img_paths:
        # Hàm load và giảm kích thước ảnh, sau đó chuyển sang bytes (cache để tăng tốc)
        @st.cache_data(show_spinner=False)
        def load_image_bytes(path):
            try:
                img = Image.open(path)
                # Giảm kích thước ảnh nếu cần (tối đa 800x800 pixel)
                img.thumbnail((800, 800))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                return buffer.getvalue()
            except Exception:
                return None

        # Tiền tải ảnh gợi ý dưới dạng bytes và lưu vào session_state
        if "loaded_images_bytes" not in st.session_state:
            loaded_bytes = []
            for path in img_paths:
                if os.path.exists(path):
                    img_bytes = load_image_bytes(path)
                    loaded_bytes.append(img_bytes)
                else:
                    loaded_bytes.append(None)
            st.session_state.loaded_images_bytes = loaded_bytes

        # In ra tất cả các ảnh gợi ý cùng lúc
        for i, img_bytes in enumerate(st.session_state.loaded_images_bytes):
            if img_bytes is not None:
                st.image(
                    img_bytes,
                    caption=f"Ảnh gợi ý {label} {i+1}/{len(img_paths)}",
                    use_container_width=True,
                )
            else:
                st.write(f"Không tìm thấy ảnh gợi ý cho ảnh {i+1}.")
    else:
        st.write("Không có ảnh gợi ý nào được tìm thấy.")
