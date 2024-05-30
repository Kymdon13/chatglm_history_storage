"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""

import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.environ.get('MODEL_PATH', '../chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    return tokenizer, model

# pickle_path = f"./chat_history/{password}.pkl"
# os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
# with open(pickle_path, "rb") as f:
#     st.session_state = pickle.load(f)
    
# pickle.dump(st.session_state, open(pickle_path, "wb"))

# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

# 初始化会话历史和过去的键值对
if "info" not in st.session_state:
    st.session_state.info = {"history_list": [], "past_key_values_list": []}

# 直接定义模型参数
max_length = 8192
top_p = 0.8
temperature = 0.6

# # 设置模型参数的滑块组
# max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
# top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
# temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)


##### 侧栏按钮
# 创建侧栏按钮的容器
if "sidebar_placeholder" not in st.session_state:
    sidebar_placeholder = st.sidebar.empty()

# 会话按钮回调函数
def callback(index):
    st.session_state.cur_id = index
    update_sidebar_buttons()
# 更新侧栏按钮状态
def update_sidebar_buttons():
    with sidebar_placeholder.container():
        try:
            st.sidebar.button("New Chat", key="new_chat", on_click=callbackForNewChat)
            st.sidebar.button("删除当前会话", key="clean", on_click=callbackForClean)
            for i, bH in enumerate(st.session_state.buttonHistorys):
                st.sidebar.button(bH, key=i, on_click=callback, args=(i,))
        except st.errors.DuplicateWidgetID:
            pass

# 初始化侧栏按钮列表、first_prompt列表、当前会话id
if 'buttonHistorys' not in st.session_state:
    st.session_state.buttonHistorys = [] # 存放按钮
if 'cur_id' not in st.session_state:
    st.session_state['cur_id'] = 0 # 用于记录当前聊天的id

# 按钮--删除当前会话
def callbackForClean():
    if len(st.session_state.buttonHistorys) == 0:
        return
    st.session_state.buttonHistorys.pop(st.session_state.cur_id)
    st.session_state.info["history_list"].pop(st.session_state.cur_id)
    st.session_state.info["past_key_values_list"].pop(st.session_state.cur_id)
    st.session_state.cur_id = 0
    update_sidebar_buttons()
    # show_history()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 按钮--新建会话
def callbackForNewChat():
    st.session_state.cur_id = len(st.session_state.buttonHistorys) + 1
    update_sidebar_buttons()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

update_sidebar_buttons()


##### 聊天框
# 每次输入后，显示历史对话
def show_history():
    try:
        for message in st.session_state.info["history_list"][st.session_state.cur_id]:
            if message["role"] == "user":
                with st.chat_message(name="user", avatar="user"):
                    st.markdown(message["content"])
            else: # message["role"] == "assistant"
                with st.chat_message(name="assistant", avatar="assistant"):
                    st.markdown(message["content"])
    except IndexError:
        pass
show_history()

# 每次输入前，显示user和assistant的头像
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty() # Placeholder for user input
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty() # Placeholder for model response

if "first_enter" not in st.session_state:
    st.session_state.first_enter = False

##### 输入并回复
prompt_text = st.chat_input("请输入您的问题😊😊😊")
if prompt_text:
    input_placeholder.markdown(prompt_text)
    try:
        history = st.session_state.info["history_list"][st.session_state.cur_id]
        print('history:',history)
        past_key_values = st.session_state.info["past_key_values_list"][st.session_state.cur_id]
        first_prompt = False if history else True
    except IndexError:
        history = []
        past_key_values = None
        first_prompt = True
    for response, history, past_key_values in model.stream_chat(
            tokenizer,
            prompt_text,
            history,
            past_key_values=past_key_values,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            return_past_key_values=True,
    ):
        message_placeholder.markdown(response)
    
    # 生成标题
    if first_prompt:
        for res, _ in model.stream_chat(
            tokenizer,
            "请你将以下的问答总结成一个5到20字的标题,问题:"+prompt_text+"答案:"+response,
            [],
            past_key_values=past_key_values,
            max_length=32768,
            top_p=0.4,
            temperature=0.2,
            return_past_key_values=False,
        ):
            pass
        # 创建侧栏按钮，创建历史对话
        st.session_state.cur_id = 0
        st.session_state.buttonHistorys.insert(0, res[:min(30, len(res))])
        st.session_state.info["history_list"].insert(0, history)
        st.session_state.info["past_key_values_list"].insert(0, past_key_values)
        # print('the length of buttonList',len(st.session_state.buttonHistorys), res)
        st.rerun()
    else:
        st.session_state.info["history_list"][st.session_state.cur_id] = history
        st.session_state.info["past_key_values_list"][st.session_state.cur_id] = past_key_values
    
    with st.chat_message(name="user", avatar="user"):
        input_placeholder = st.empty() # Placeholder for user input
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.empty() # Placeholder for model response
