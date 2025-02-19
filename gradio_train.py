import gradio as gr
import transformers
from transformers import pipeline

# Interface是Gradio的一个高级组件，它支持一定的自定义内容，但是其灵活性略差

# def to_black(image):
#     output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return output
#
#
# interface = gr.Interface(fn=to_black, inputs="image", outputs="image",
#                          examples=[["C:\\Users\\16354\\Pictures\\1739097969270.jpg"]])
# interface.launch()
# interface.launch(share=True)


## 基于Pipeline方法
# 搭建阅读理解应用程序
# 搭建一个文本分类应用程序。这里使用的模型是来自Hugging Face网站的uer/roberta- base-finetuned-dianping-chinese。
# gr.Interface.from_pipeline(
#     pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()
# title = "抽取式问答"
# description = "输入上下文与问题后，点击submit按钮，可从上下文中抽取出答案！"
# examples = [
#     [
#         "李白，字太白，号青莲居士，祖籍陇西成纪（今甘肃省秦安县），出生于蜀郡绵州昌隆县（一说出生于西域碎叶）。唐朝伟大的浪漫主义诗人，代表作有《望庐山瀑布》《行路难》《蜀道难》《将进酒》《早发白帝城》等。",
#         "《早发白帝城》的作者是"],
#     [
#         "李白为人爽朗大方，乐于交友，爱好饮酒作诗，名列酒中八仙。曾经得到唐玄宗李隆基赏识，担任翰林供奉，赐金放还，游历全国。李白所作词赋，就其开创意义及艺术成就而言，享有极为崇高的地位，后世誉为诗仙，与诗圣杜甫并称李杜。",
#         "被誉为诗仙的人是"]
# ]
# article = "大模型开发基础"
# gr.Interface.from_pipeline(
#     transformers.pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa"),
#     title=title, description=description, examples=examples, article=article).launch()

##自定义实现方式

# 要求输入包含上下文背景(Context)和问题(Question)两部分，输出包含回答(Answer)和得分(Score)两部分。
# 此处的回答内容需要与问题进行拼接，形式为“问题:回答”。

# 定义custom_predict()方法。该方法的输入包括Context和Question两部分，输出包括Answer和Score两部分，
# 本质上还是调用Pipeline方法进行推理，但是在生成答案时进行了额外的拼接处理

# qa = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
#
#
# def custom_predict(context, question):
#     answer_result = qa(context=context, question=question)
#     answer = question + ": " + answer_result["answer"]
#     score = answer_result["score"]
#     return answer, score


# 在Interface()方法中绑定执行方法并指定输入输出组件。其中，参数fn与custom_predict()方法进行绑定；
# 参数inputs指定输入组件，因为本应用中包含context和question两个文本类型的输入，
# 所以参数inputs的值为["text","text"]（text表示输入组件为TextBox，text只是一种便捷的指定方式）；
# 参数outputs指定输出组件，answer是文本类型的输出，score是标签类型的输出，这里采用了与inputs字段不一样的创建方式，
# 可以直接创建对应的组件。这种方式的优势在于可以对组件进行精细配置，例如此处分别指定了两个输出模块的label。
# gr.Interface(fn=custom_predict,
#              inputs=["text", "text"],
#              outputs=[gr.Textbox(label="answer"),
#                       gr.Label(label="score")],
#              title=title,
#              description=description,
#              examples=examples,
#              article=article).launch()


## Blocks是比Interface更加底层的模块，它支持简单的自定义排版。

title = "抽取式问答"
description = "输入上下文与问题后，点击submit按钮，可从上下文中抽取出答案！"
examples = [
    [
        "李白，字太白，号青莲居士，祖籍陇西成纪（今甘肃省秦安县），出生于蜀郡绵州昌隆县（一说出生于西域碎叶）。唐朝伟大的浪漫主义诗人，代表作有《望庐山瀑布》《行路难》《蜀道难》《将进酒》《早发白帝城》等。",
        "《早发白帝城》的作者是"],
    [
        "李白为人爽朗大方，乐于交友，爱好饮酒作诗，名列酒中八仙。曾经得到唐玄宗李隆基赏识，担任翰林供奉，赐金放还，游历全国。李白所作词赋，就其开创意义及艺术成就而言，享有极为崇高的地位，后世誉为诗仙，与诗圣杜甫并称李杜。",
        "被誉为诗仙的人是"]
]
article = "大模型开发基础"
qa = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")


def custom_predict(context, question):
    answer_result = qa(context=context, question=question)
    answer = question + ": " + answer_result["answer"]
    score = answer_result["score"]
    return answer, score


def clear_input():
    return "", "", "", ""


with gr.Blocks() as demo:
    gr.Markdown("# 抽取式问答")
    gr.Markdown("输入上下文与问题后，点击submit按钮，可从上下文中抽取出答案！")
    with gr.Column():
        context = gr.Textbox(label="context")
        question = gr.Textbox(label="question")
    with gr.Row():  # 行排列
        clear = gr.Button("clear")
        submit = gr.Button("submit")
    with gr.Column():  # 列排列
        answer = gr.Textbox(label="answer")
        score = gr.Label(label="score")
    # 绑定submit点击方法
    submit.click(fn=custom_predict, inputs=[context, question], outputs=[answer, score])
    # 绑定clear点击方法
    clear.click(fn=clear_input, inputs=[], outputs=[context, question, answer, score])
    gr.Examples(examples, inputs=[context, question])
    gr.Markdown("大模型开发")
demo.launch()
