1. Data preparation and model fitting

The purpose of this project was to create a model that would predict whether a flight would arrive on time or be delayed based on choice of airline and how busy the departure and arrival airports are on an annual basis.

The data used as part of this was found on the Kaggle website and extracted from data that I had previously compiled during my normal course of work.  Information about the Kaggle data can be found in the &quot;Project Concept.md&quot; markdown document in this folder (or the &quot;Project Concept.docx&quot; file).  The relevant data that I provided consisted of airport operations count information that was a direct measure of how busy each airport is.

The &quot;Flight Delay Model.ipynb&quot; Jupyter notebook file is where the data wrangling logic can be found.  Flight, airport, and airline information was loaded.  Canceled and diverted flight information was removed.  Rows with NULL data, missing data, and invalid Airport identifier information were removed.

After the initial cleaning, the airport, airline and operations data sets were merged.  Counts were calculated for the various airlines and one-hot encoding was performed for the airlines with the higher late flight counts.  This was used to help decide which columns to use for the model.

A logistic regression model was employed, fitted, and a pickle file was created to serialize this model to disk.  The pickle file, named FlightDelayLogRegModel.pik, is included in this directory.

The inputs to this model consist of the origin airport annual operation count, the destination airport annual operation count, and which of the 6 &quot;tardiest&quot; airlines was selected for the respective flight.

1. Flask Deployment

The Flask environment within the Visual Studio 2019 IDE was used to emulate deployment.

The runserver.py file contained in this directory houses the logic used to create the &#39;predict&#39; endpoint used to facilitate a web-based REST method to use the model previously serialized into a pickle file to predict whether a flight will be delayed or not.  It also provides a &#39;health&#39; endpoint that can be used to ensure the interface is active by the end user.

To run the project, start the project in Visual Studio 2019.  This will display a main page shown below:

Next, open a command window (if in a Windows environment).  Below is a sample curl command to call the prediction endpoint and the response from the model contained in the pickle file.  In this example, the prediction returned &#39;0&#39;, indicating that the flight would not be late (i.e., on time) and the prediction probabilities.

curl -d &quot;{\&quot;OpsOrigin\&quot;:\&quot;30000\&quot;, \&quot;OpsDest\&quot;:\&quot;50000\&quot;, \&quot;AA\&quot;:\&quot;1\&quot;, \&quot;DL\&quot;:\&quot;0\&quot;, \&quot;EV\&quot;:\&quot;0\&quot;, \&quot;OO\&quot;:\&quot;0\&quot;, \&quot;UA\&quot;:\&quot;0\&quot;, \&quot;WN\&quot;:\&quot;0\&quot;}&quot; -H &quot;Content-Type: application/json&quot; -X POST http://localhost:54438/predict