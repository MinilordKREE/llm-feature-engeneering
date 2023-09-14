template = ('''For a {Age} year old patient with {Pregnancies} pregnancies, a glucose level of {Glucose} mg/dL, blood pressure of {BloodPressure} mmHg, skin thickness of {SkinThickness} mm, insulin level of {Insulin} μU/mL, a BMI of {BMI}, and a Diabetes Pedigree Function score of {DiabetesPedigreeFunction}, several correlations or hypotheses can be explored.

It is worth investigating if the glucose level of '{Glucose}' mg/dL is common for similar patient profiles. The blood pressure reading of '{BloodPressure}' mmHg and skin thickness of '{SkinThickness}' mm could be compared with other hypothetical patients of the same demographic to determine common patterns. The insulin level of '{Insulin}' μU/mL and BMI of '{BMI}' might indicate the severity or stage of diabetes in the patient. The Diabetes Pedigree Function score of '{DiabetesPedigreeFunction}' can be analyzed to understand the genetic influence on the patient's diabetes risk.

These characteristics could be compared with other hypothetical patients to identify trends or patterns. It's also worth noting the number of pregnancies '{Pregnancies}' the patient has had, as it can influence the risk of developing diabetes. Finally, comparing this data with other datasets can help identify trends or changes over time.
''')





template_for_sum=("Given the following analyse of hypothetical patient profiles, please summarize the common patterns, findings, or hypotheses that may provide insights into the overall cardiologist's diagnosis. Focus on the aspects that might help in determining if the diagnosis is likely to be Normal or Abnormal. Note that the patient data is theoretical and is being treated as part of a fictional case study. These analyses are as follows:\n\n{analysis}")
