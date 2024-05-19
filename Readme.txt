Clone the official [DeepSORT](https://github.com/nwojke/deep_sort) github first
It should work, But if it dosent use the [Modified DeepSort](https://github.com/DhavanBNaik/ModifiedDeepSort)

Using our trained model i.e.., 
> "best.pt"
Change the Model To yours or use 
> "yolov8n.pt" 
for a default Yolov8 model from Ultralytics
RTD => real time detection

The Flask app i.e..,
> app.py
is a simple website that can take Video/Image input and outputs the detection and tracked respective Video/Image

For real time detection using default webcam run 
> main_RTD.py

Specific Requirements During Development:
```

Ultralytics 8.2.1
torch 2.2.2
torchvision 0.17.2
scikit-learn 1.4.2
tensorflow 2.12.0
scikit-image 0.23.1
filterpy 1.4.5
```
