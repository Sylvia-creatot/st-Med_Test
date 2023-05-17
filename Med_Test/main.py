import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import os
from PIL import Image
from User_Data import collect_and_store_data

#st.set_page_config(page_title="Med-Test")
    
l1=['Vaginal discharge','Burning sensation','Sexpain',
    'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    'Genital warts']

disease = ['Female Chlamydia','Female Gonorrhoea', 'Female Trichomoniasis',
            'Genital Herpes', 'HPV', 'Syphilis']

df = pd.read_csv("Training2.csv")

df.replace({'prognosis':{"Female Chlamydia":0,'Female Gonorrhoea':1,
            'Female Trichomoniasis':2,'Genital Herpes':3,'HPV':4,'Syphilis':5}}, inplace=True)

X = df[l1]

y = df[['prognosis']]
np.ravel(y)

tf = pd.read_csv("Testing.csv")

tf.replace({'prognosis':{"Female Chlamydia":0,'Female Gonorrhoea':1,
            'Female Trichomoniasis':2,'Genital Herpes':3,'HPV':4,'Syphilis':5}}, inplace=True)

X_test = tf[l1]
y_test = tf[["prognosis"]]
np.ravel(y_test)

def doMyTask2(data):
	from sklearn import tree
	l2=[]
	for x in range(0,len(l1)):
		l2.append(0)
	clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
	clf3 = clf3.fit(X,y)

	# calculating accuracy-------------------------------------------------------------------
	from sklearn.metrics import accuracy_score
	y_pred=clf3.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred,normalize=False))
	# -----------------------------------------------------

	psymptoms = [data[0],data[1],data[2],data[3],data[4]]

	for k in range(0,len(l1)):
		# print (k,)
		for z in psymptoms:
			if(z==l1[k]):
				l2[k]=1

	inputtest = [l2]
	predict = clf3.predict(inputtest)
	predicted=predict[0]

	h='no'
	for a in range(0,len(disease)):
		if(predicted == a):
			h='yes'
			break

	#result = str(disease[predicted])
	score = round(accuracy_score(y_test, y_pred),2)

	st.subheader("Results")
	#st.markdown("With an accuracy of "+ str(accuracy_score(y_test, y_pred)))
	st.markdown("With an accuracy of "+ str(score))
	st.markdown("You may be suffering from **" + disease[predicted]+"**")
	#print(type(disease[predicted]))

	return disease[predicted]

def doMyTask(data):

	from sklearn.ensemble import RandomForestClassifier
	l2=[]
	for x in range(0,len(l1)):
		l2.append(0)
	clf4 = RandomForestClassifier()
	clf4 = clf4.fit(X,np.ravel(y))

	# calculating accuracy-------------------------------------------------------------------
	from sklearn.metrics import accuracy_score
	y_pred=clf4.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred,normalize=False))
	# -----------------------------------------------------

	psymptoms = [data[0],data[1],data[2],data[3],data[4]]

	for k in range(0,len(l1)):
		for z in psymptoms:
			if(z==l1[k]):
				l2[k]=1

	inputtest = [l2]
	predict = clf4.predict(inputtest)
	predicted=predict[0]

	h='no'
	for a in range(0,len(disease)):
		if(predicted == a):
			h='yes'
			break

	#result = str(disease[predicted])
	score = round(accuracy_score(y_test, y_pred),2)

	st.subheader("Results")
	#st.markdown("With an accuracy of "+ str(accuracy_score(y_test, y_pred)))
	st.markdown("With an accuracy of "+ str(score))
	st.markdown("You may be suffering from **" + disease[predicted]+"**")
	#print(type(disease[predicted]))

	return disease[predicted]

def RUN():
	st.sidebar.title("Med-Test")

	app_choice = st.sidebar.selectbox("Go to",
					['Home',"Self_test",])
	if app_choice == "Home":
		home()

	elif app_choice == "Self_test":
		Self_test()
def home():
    st.title("....")

    # Add the logo
image = Image.open("C:/Users/Project/Med_Test/images/image2.jpg")

# Display image
st.image(image, width=700)

    # Add the description
desc_text = """ (Image by: Topline MD.)

Welcome to Med-Test!! This is a medical web app that is designed to assist you self-diagnose yourself on the type of STD you may have contracted."""
st.markdown(f"{desc_text}")

    # survey request
survey= """ """
st.markdown(survey)
st.markdown("<h2 style='color: purple;'>After self_test please participate in the survey below. It takes less than one minute to complete!!!!.</h2>", unsafe_allow_html=True)



    # contents of the self_test page
def Self_test():
	st.title("Self Test Page")
	symptom1 = st.selectbox("Symptom 1",['','Vaginal discharge','Burning sensation','Sexpain',
    						'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    						'Genital warts'])

	symptom2 = st.selectbox("Symptom 2",['','Vaginal discharge','Burning sensation','Sexpain',
    						'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    						'Genital warts'])
	
	symptom3 = st.selectbox("Symptom 3",['','Vaginal discharge','Burning sensation','Sexpain',
    						'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    						'Genital warts'])

	symptom4 = st.selectbox("Symptom 4",['','Vaginal discharge','Burning sensation','Sexpain',
    						'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    						'Genital warts'])
	
	symptom5 = st.selectbox("Symptom 5",['','Vaginal discharge','Burning sensation','Sexpain',
    						'Mouth warts','Rash','Vaginal bleeding','Genital itchiness', 'Genital blisters',
    						'Genital warts'])

	col1, col2 = st.columns([1,1])
	with col1:
		treeClass = st.button("Test")
	
	#with col2:
		#randForest = st.button("Random Forest")

	#submit = st.button("Submit")

	if treeClass:
		doMyTask2([symptom1,symptom2,symptom3,symptom4,symptom5])

	#elif randForest:
		#doMyTask([symptom1,symptom2,symptom3,symptom4,symptom5])		
		
		# Call the function to execute the code
collect_and_store_data()


RUN()