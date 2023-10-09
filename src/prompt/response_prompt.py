# template = ('''For a {Age} year old patient with {Pregnancies} pregnancies, a glucose level of {Glucose} mg/dL, blood pressure of {BloodPressure} mmHg, skin thickness of {SkinThickness} mm, insulin level of {Insulin} μU/mL, a BMI of {BMI}, and a Diabetes Pedigree Function score of {DiabetesPedigreeFunction}, several correlations or hypotheses can be explored.

# It is worth investigating if the glucose level of '{Glucose}' mg/dL is common for similar patient profiles. The blood pressure reading of '{BloodPressure}' mmHg and skin thickness of '{SkinThickness}' mm could be compared with other hypothetical patients of the same demographic to determine common patterns. The insulin level of '{Insulin}' μU/mL and BMI of '{BMI}' might indicate the severity or stage of diabetes in the patient. The Diabetes Pedigree Function score of '{DiabetesPedigreeFunction}' can be analyzed to understand the genetic influence on the patient's diabetes risk.

# These characteristics could be compared with other hypothetical patients to identify trends or patterns. It's also worth noting the number of pregnancies '{Pregnancies}' the patient has had, as it can influence the risk of developing diabetes. Finally, comparing this data with other datasets can help identify trends or changes over time.
# ''')

# template_eucaly = ('''For a eucalyptus tree from {Abbrev} with a representative value of {Rep}, located in {Locality} (Map Reference: {Map_Ref}, Latitude: {Latitude}), at an altitude of {Altitude} meters, and receiving {Rainfall} mm of rainfall, several correlations or hypotheses can be explored.

# It is worth investigating if the altitude of '{Altitude}' meters is common for trees of similar profiles. The rainfall of '{Rainfall}' mm and the number of frosts '{Frosts}' could be compared with other eucalyptus trees from the same or different localities to determine common patterns. The year '{Year}' and species '{Sp}' might indicate specific environmental or genetic factors influencing the tree's growth and health.

# The tree's physical characteristics such as PMC number '{PMCno}', diameter at breast height (DBH) of '{DBH}' cm, height '{Ht}' meters, survival rate '{Surv}', vigor '{Vig}', internal resistance '{Ins_res}', stem form '{Stem_Fm}', crown form '{Crown_Fm}', and branch form '{Brnch_Fm}' can be analyzed to understand its overall health and adaptability.

# These characteristics could be compared with other eucalyptus trees to identify trends or patterns. Comparing this data with other datasets can help identify trends or changes over time.
# ''')



# template_for_sum=("Given the following analyse of hypothetical patient profiles, please summarize the common patterns, findings, or hypotheses that may provide insights into the overall cardiologist's diagnosis. Focus on the aspects that might help in determining if the diagnosis is likely to be Normal or Abnormal. Note that the patient data is theoretical and is being treated as part of a fictional case study. These analyses are as follows:\n\n{analysis}")






prompt_tempelate=("""
I'm crafting a contextual narrative to analyze a specific data point from a dataset.        
                
Given a dataset related to:
{Data Description}. 

With the record:
{Data Point}

and schema:
{Schema}

I will create a narrative focusing on the outcome variable {Outcome}, adhering to the following constraints:
1. The narrative will be concise, not exceeding 350 words.
2. It will be highly relevant to {Outcome}, providing insights or context about how the variables in the specific record might relate to or impact {Outcome}.

Here's my analysis:
""")


prompt_template_circor=("""
I'm crafting a contextual narrative to analyze a specific data point from a dataset.

Given a dataset related to:
The CirCor DigiScope Phonocardiogram Dataset, which encompasses 5272 heart sound recordings from 1568 subjects aged between 0 and 21 years. The dataset, notable for being the largest publicly available pediatric heart sound dataset, is utilized in the George B. Moody PhysioNet Challenge 2022 on Heart Murmur Detection from Phonocardiogram Recordings, focusing on text-based data rather than audible files.

With the record:
A single record might represent a subject of Age category '{Age}', Sex '{Sex}', Height '{Height}' cm, Weight '{Weight}' kg, Pregnancy status '{Pregnancy_status}', Murmur '{Murmur}', and various binary indicators for murmur locations (e.g., PV '{PV}', TV '{TV}', AV '{AV}', MV '{MV}'), Most audible location '{Most_audible_location}', and several encoded murmur characteristics (e.g., Systolic murmur timing '{Systolic_murmur_timing}', Systolic murmur shape '{Systolic_murmur_shape}', etc.), with an Outcome '{Outcome}' and Campaign '{Campaign}'.

and schema:
The schema includes variables such as Age (encoded categorical: [Neonate, Infant, Child, Adolescent, Young adult]), Sex (binary: [Female, Male]), Height (continuous: > 0), Weight (continuous: > 0), Pregnancy status (binary: [True, False]), Murmur (encoded categorical: [Present, Absent, Unknown]), PV, TV, AV, MV (binary: [True, False]), Most audible location (encoded categorical: [PV, TV, AV, MV, Phc]), various encoded murmur characteristics (categorical), Outcome (binary: [Normal, Abnormal]), and Campaign (binary: [CC2014, CC2015]).

I will create a narrative focusing on the outcome variable {Outcome}, adhering to the following constraints:
1. The narrative will be concise, not exceeding 350 words.
2. It will be highly relevant to {Outcome}, providing insights or context about how the variables in the specific record might relate to or impact {Outcome}.

Here's my analysis:
""")

prompt_tempelate_diabetes=("""
I'm tasked with crafting a contextual narrative to analyze a dataset.

Given a dataset related to:
{Data Description}. 

With a representative record:
{Data Point}

and schema:
{Schema}

I will create a narrative focusing on the outcome variable {Outcome}, ensuring that:
1. The narrative is concise, not exceeding 350 words.
2. It is highly pertinent to {Outcome}, exploring its relationships with other variables in the dataset and providing insights into patterns or trends observed.

In light of the above, my analysis is as follows:
""")

prompt_tempelate_euca=("""
I'm tasked with crafting a contextual narrative to analyze a dataset.

Given a dataset related to:
{Data Description}. 

With a representative record:
{Data Point}

and schema:
{Schema}

I will create a narrative focusing on the outcome variable {Outcome}, ensuring that:
1. The narrative is concise, not exceeding 350 words.
2. It is highly pertinent to {Outcome}, exploring its relationships with other variables in the dataset and providing insights into patterns or trends observed.

In light of the above, my analysis is as follows:
""")


prompt_tempelate_heart_disease=("""
I'm tasked with crafting a contextual narrative to analyze a dataset.

Given a dataset related to:
{Data Description}. 

With a representative record:
{Data Point}

and schema:
{Schema}

I will create a narrative focusing on the outcome variable {Outcome}, ensuring that:
1. The narrative is concise, not exceeding 350 words.
2. It is highly pertinent to {Outcome}, exploring its relationships with other variables in the dataset and providing insights into patterns or trends observed.

In light of the above, my analysis is as follows:
""")