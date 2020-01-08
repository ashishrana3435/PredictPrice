from django.shortcuts import render,HttpResponse

import numpy as np
from sklearn.preprocessing import StandardScaler

import random
# Create your views here.

def main(request):
    return HttpResponse("HELLO THERE <a href='/index'>Index</a>")
def index(request):
    number = "Ashish"
    if(request.method == "POST"):
        form = request.POST
        bedroom = form['Bedrooms']
        bathroom = form['Bathrooms']
        area = form['Area']
        waterfront = form['Waterfront']
        b = np.array([-1.94571236e-16, -1.20610776e-01,  5.16206330e-02,  7.03696660e-01,1.89915143e-01])
        X = np.array([1,bedroom,bathroom,area,waterfront])
        sc = StandardScaler()
        X = sc.fit_transform(X.reshape(-1,1))
        X = X.T
        pred = np.dot(X,b)
        final_pred = sc.inverse_transform(pred)
        print(final_pred)
        context = {
            'Bathrooms': bathroom,
            'Bedrooms': bedroom,
            'Area': area,
            'Waterfront': waterfront,
            'Prediction': final_pred[0],
        }
        return render(request, "output.html", context)
    return render(request, "index.html", {'number':number})
