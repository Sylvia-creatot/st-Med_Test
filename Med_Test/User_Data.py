import csv
import streamlit as st

def collect_and_store_data():
    # County names
    counties = ['Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 'Taita Taveta', 'Garissa', 'Wajir', 'Mandera', 'Marsabit','Isiolo',
                'Meru','Tharaka-Nithi','Embu','Kitui','Machakos','Makueni','Nyandarua','Nyeri','Kirinyaga','Murang’a','Kiambu ','Turkana',
                'West Pokot', 'Samburu','Trans-Nzoia','Uasin Gishu','Elgeyo-Marakwet','Nandi','Baringo','Laikipia',
                'Nakuru','Narok','Kajiado','Kericho','Bomet','Kakamega','Vihiga','Bungoma','Busia','Siaya','Kisumu','Homa Bay','Migori', 
                'Kisii','Nyamira','Nairobi'] 

    # Diseases
    diseases = ['Female Chlamydia','Female Gonorrhoea', 'Female Trichomoniasis',
            'Genital Herpes', 'HPV', 'Syphilis']  

    # Initialize variables to store user selections
    county_selection = None
    disease_selection = None

    # Create the Streamlit app
    st.title("Survey")

    # Display the county selection dropdown
    county_selection = st.selectbox("Select County", counties)

    # Display the disease selection dropdown
    disease_selection = st.selectbox("Select Disease", diseases)

    # Save data to CSV file
    if st.button("Submit"):
        # Open the CSV file
        with open('disease_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            # Write the headers if the file is empty
            if file.tell() == 0:
                headers = ['County'] + diseases
                writer.writerow(headers)

            # Write the data to the CSV file
            data = [county_selection] + [0] * len(diseases)  # Initialize all disease counts as 0
            disease_index = diseases.index(disease_selection)
            data[disease_index + 1] = 1  # Set the selected disease count as 1
            writer.writerow(data)

        st.success("All done!!.")
