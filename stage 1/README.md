### Языковая модель для поиска ответов по текстам, используется RAG для LLM
Модель: Saiga 13b
Gradio для интерфейса
travelline-data.zip - содержимое онлайн базы знаний https://www.travelline.ru/support/knowledge-base/
LLM-RAG.pdf - внешний вид интерфейса с настроечными элементами

для тестирования использовался VDS 1xV100/16GB 10VCPU/28Гб

pip install -r requirements.txt
python app.py
http://0.0.0.0:7860