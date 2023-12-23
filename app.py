from fastai.vision.all import *  # noqa: F403
from fastai.vision.widgets import *  # noqa: F403
import gradio as gr

# Load the pre-trained model, the model has been trained on the following Butterflies:
# 1. Monarch
# 2. Painted Lady
# 3. Red Admiral
# 4. Viceroy
# 5. Bronze Copper
# 6. Buckeye
# The model has an error_rate of ~6.7%
learn = load_learner('kinds_of_butterflies_model.pkl')


def classify_image(img):
  pred, idx, probs = learn.predict(img)
  categories = learn.dls.vocab
  label_pred = widgets.Label()
  label_pred.value = f'Prediction: {pred}; Probability: {probs[idx]:.04f}'
  print(label_pred)
  float_values = map(float, probs)
  return dict(zip(categories, float_values))


image = gr.Image()
label = gr.Label()
examples = ['forest.jpg', 'monarch.jpeg', 'swallow.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
