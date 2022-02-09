# DeepFake-Detection-Using-Pytorch
 
 
 

<h1 color="green"><b>Deep Fake Detection Package</b></h1>

---

<h1 color="green"><b>Abstract</b></h1>
<p>This project Deepfakes can distort our perception of the truth and we need to develop a strategy to improve their detection. Deep Fakes are increasingly detrimental to privacy, social security, and democracy. We plan to achieve better accuracy in predicting real and fake videos..</p>


<h1 color="green"><b>Instructions to Install our Deep Fake Detection Package</b></h1>


1. Install:

```python
pip install Deep-Fake-Detection
```

2. Import the DeepFake_Utils from deepfake_detection :

```python
from deepfake_detection import DeepFake_Utils
from moviepy.editor import *
```

3. Detect Detect the video is fake or not by Ensemble ResNext and Inception Custom Models :

```python
# Run the Below Function by Input your Video Path to get the outPutVideo with Label Fake on Real on it
DeepFake_Utils.Inference_on_video(OutputVideoPath,InputVideoPath)


# Show the Video
VideoFileClip(OutputVideoPath, audio=False, target_resolution=(300,None)).ipython_display()
```
