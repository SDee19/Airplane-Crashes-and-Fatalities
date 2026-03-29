![360_F_893672169_CAVy2imJYhoZKPHOHE0fB91fgSUjtKP8](https://github.com/user-attachments/assets/794e9a33-ca7e-4915-a1d6-ef3df040f05c)

# Airplane-Crashes-and-Fatalities

The aviation industry has a storied history marked by significant advancements in safety, yet airplane crashes remain a critical area of study for improving future outcomes. By examining features such as aircraft type, operator, time period, and crash causes, and introducing a derived feature, "𝗦𝘂𝗿𝘃𝗶𝘃𝗮𝗹 𝗦𝗲𝘃𝗲𝗿𝗶𝘁𝘆" as the response, this study uncovers patterns and relationships that contribute to survival outcomes. The analysis employs visualizations and statistical methods to explore temporal trends, operator differences, and crash cause impacts on survival rates. These insights are offering aviation stakeholders, researchers, and the public a tool to explore historical crash data and inform safety enhancements, ultimately contributing to safer air travel.

The core of this work is a machine learning model that analyzes historical crash data to identify subtle patterns influencing survival outcomes. The insights from this model are then utilized to create "𝗦𝗸𝘆𝗦𝗮𝗳𝗲", a targeted data product designed for Safety Officers, Engineers, and Executives. SkySafe provides these professionals with a powerful tool to move beyond reactive study. By enabling them to quantitatively validate the anticipated impact of their proposed plans on survival outcomes, the platform allows for data-driven decision-making that will demonstrably enhance safety and contribute to the continuous mitigation of risk across the industry. 

# Description of the Question

"𝗜𝘀 𝘀𝘂𝗿𝘃𝗶𝘃𝗮𝗹 𝘂𝗻𝗶𝗳𝗼𝗿𝗺 𝗮𝗰𝗿𝗼𝘀𝘀 𝗰𝗿𝗮𝘀𝗵𝗲𝘀, 𝗼𝗿 𝗱𝗼 𝘀𝗼𝗺𝗲 𝗹𝗼𝗰𝗮𝘁𝗶𝗼𝗻𝘀 𝗮𝗻𝗱 𝗮𝗶𝗿𝗹𝗶𝗻𝗲𝘀 𝗼𝗳𝗳𝗲𝗿 𝗯𝗲𝘁𝘁𝗲𝗿 𝗼𝘂𝘁𝗰𝗼𝗺𝗲𝘀?"

One of the key challenges in aviation safety analysis is understanding the factors that influence survival rates in airplane crashes, a critical metric for assessing the effectiveness of safety protocols and emergency responses. The myth that "𝗧𝗵𝗲 𝗹𝗼𝗰𝗮𝘁𝗶𝗼𝗻 𝗼𝗿 𝗼𝗽𝗲𝗿𝗮𝘁𝗼𝗿 𝗼𝗳 𝘁𝗵𝗲 𝗳𝗹𝗶𝗴𝗵𝘁 𝗱𝗼𝗲𝘀𝗻'𝘁 𝗮𝗳𝗳𝗲𝗰𝘁 𝘀𝘂𝗿𝘃𝗶𝘃𝗮𝗹—𝗮𝗹𝗹 𝗰𝗿𝗮𝘀𝗵𝗲𝘀 𝗮𝗿𝗲 𝗲𝗾𝘂𝗮𝗹𝗹𝘆 𝗱𝗲𝗮𝗱𝗹𝘆" suggests a uniform distribution of fatality outcomes, regardless of geography, airline, or other contextual factors, which is a subtle yet rarely questioned assumption. This 
oversimplification challenges the idea that survival is purely luck-based and ignores the potential impact of variables such as location entity, operator protocols, regional infrastructure, and temporal factors. To address this challenge, a statistical approach is crucial for uncovering patterns and relationships that can inform safety improvements. Therefore, the initial objective of this project is to identify key factors influencing survival across crashes through analyzing variations in survival outcomes by location, operator, and time. 

Building upon our initial objective, a new and more complex problem emerges: 

"𝗧𝗵𝗲 𝗶𝗻𝗱𝘂𝘀𝘁𝗿𝘆'𝘀 𝗶𝗻𝗮𝗯𝗶𝗹𝗶𝘁𝘆 𝘁𝗼 𝘁𝗿𝗮𝗻𝘀𝗳𝗼𝗿𝗺 𝘁𝗵𝗲 𝗵𝗶𝘀𝘁𝗼𝗿𝗶𝗰𝗮𝗹 𝗶𝗻𝘀𝗶𝗴𝗵𝘁𝘀 𝗶𝗻𝘁𝗼 𝗮 𝗽𝗿𝗼𝗮𝗰𝘁𝗶𝘃𝗲, 𝗳𝗼𝗿𝘄𝗮𝗿𝗱-𝗹𝗼𝗼𝗸𝗶𝗻𝗴 𝗰𝗮𝗽𝗮𝗯𝗶𝗹𝗶𝘁𝘆. 𝗔𝘃𝗶𝗮𝘁𝗶𝗼𝗻 𝗽𝗿𝗼𝗳𝗲𝘀𝘀𝗶𝗼𝗻𝗮𝗹𝘀—𝗳𝗿𝗼𝗺 𝗦𝗮𝗳𝗲𝘁𝘆 𝗢𝗳𝗳𝗶𝗰𝗲𝗿𝘀 𝘁𝗼 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝘀 𝗮𝗻𝗱 𝗘𝘅𝗲𝗰𝘂𝘁𝗶𝘃𝗲𝘀—𝗰𝘂𝗿𝗿𝗲𝗻𝘁𝗹𝘆 𝗹𝗮𝗰𝗸 𝗮 𝗾𝘂𝗮𝗻𝘁𝗶𝘁𝗮𝘁𝗶𝘃𝗲 𝘁𝗼𝗼𝗹 𝘁𝗼 𝗺𝗼𝘃𝗲 𝗯𝗲𝘆𝗼𝗻𝗱 𝗿𝗲𝘁𝗿𝗼𝘀𝗽𝗲𝗰𝘁𝗶𝘃𝗲 𝗮𝗻𝗮𝗹𝘆𝘀𝗶𝘀"

The core challenge is the absence of a system that can effectively assess and predict the survival severity of their proposed safety enhancements, whether they are new aircraft designs or revised maintenance protocols. This critical gap prevents truly data-driven decision-making and underscores the need for a solution that empowers the industry to mitigate risk before an incident occurs, rather than simply reacting to past events. 

# Description of the Dataset

The "𝗔𝗶𝗿𝗽𝗹𝗮𝗻𝗲 𝗖𝗿𝗮𝘀𝗵𝗲𝘀 𝗮𝗻𝗱 𝗙𝗮𝘁𝗮𝗹𝗶𝘁𝗶𝗲𝘀 𝗦𝗶𝗻𝗰𝗲 𝟭𝟵𝟬𝟴" dataset, sourced from Kaggle.com, comprises a dataset of over 5,000 records, each detailing an individual "𝗕𝗼𝗲𝗶𝗻𝗴 𝟳𝟬𝟳" airplane crash incident. These records include 13 distinct variables, encompassing a range of crash-related metrics, and provide a comprehensive historical overview of aviation incidents from 1908 to 2003. 

<img width="1190" height="470" alt="Screenshot 2026-03-29 101649" src="https://github.com/user-attachments/assets/458331b7-a2db-4b64-acc5-8593abbbdbd2" />

𝗟𝗶𝗻𝗸 𝗳𝗼𝗿 𝘁𝗵𝗲 𝗱𝗮𝘁𝗮𝘀𝗲𝘁: https://www.kaggle.com/datasets/thedevastator/airplane-crashes-and-fatalities/data


# Conclusions and Discussion

1. 𝗘𝘅𝗽𝗹𝗼𝗿𝗮𝘁𝗼𝗿𝘆 𝗗𝗮𝘁𝗮 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀 (𝗘𝗗𝗔)

   The descriptive analysis offers compelling and diverse evidence that both a flight's location and its operator significantly influence survival outcomes. This demonstrates that survival is not uniform across all crashes; rather, the severity of outcomes is shaped by a confluence of factors, including environmental conditions, geography, infrastructure, emergency response time, the type of operator, and historical safety practices. Consequently, the widely held belief that all crashes are equally deadly is hereby 𝗱𝗲𝗯𝘂𝗻𝗸𝗲𝗱, as survival demonstrably varies meaningfully based on location and operator. 

2. 𝗔𝗱𝘃𝗮𝗻𝗰𝗲𝗱 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀
   
   This advanced analysis marks a pivotal transition from a retrospective understanding of aviation incidents to a proactive, predictive capability. By moving beyond our descriptive findings, we have developed a data product, SkySafe, designed to empower aviation professionals with actionable insights to mitigate risk before it escalates.

   Our methodology, which included a rigorous comparison of two distinct analytical approaches, led to several key conclusions. First, the "𝗫𝗚𝗕𝗼𝗼𝘀𝘁" classifier consistently emerged as the superior model for predicting "𝗦𝘂𝗿𝘃𝗶𝘃𝗮𝗹 𝗦𝗲𝘃𝗲𝗿𝗶𝘁𝘆", outperforming all other models with its high predictive recall. This validated our decision to prioritize this metric, as it directly addresses the critical need to minimize false negatives—the most dangerous error in aviation safety prediction.

   Second, the comparative analysis of our two approaches—one using the initial set of 17 predictors and the other using dimensionality-reduced components—provided a definitive path forward. The first approach, when refined to include only the eight most important predictors, not only achieved a higher "𝗿𝗲𝗰𝗮𝗹𝗹" (0.9669) but also demonstrated a significant reduction in false negatives. This finding is crucial, as it confirms that a more focused, parsimonious model can yield more accurate and reliable results.

   Furthermore, the interpretability of our optimal model proved to be a powerful validation of our entire analysis. By examining the partial dependence plots of the key predictor variables, we confirmed that the model's internal logic aligns perfectly with both our initial EDA findings and established aviation safety research. This provides a high degree of confidence in the model's predictions, ensuring that the insights generated by SkySafe are both statistically robust and contextually sound.

   Despite these successes, the analysis highlighted several limitations. The persistent challenge of overfitting, even after extensive hyperparameter tuning, indicates a need for continued model refinement. More importantly, the reliance on a historical dataset spanning from 1908 to 2003 presents a critical limitation for present-day applications. While the model's high recall suggests its foundational strength, its projections must be viewed with the understanding that they are not based on the most recent technological and regulatory advancements. 

In conclusion, this project successfully developed and validated a predictive model that serves as the foundation for the "𝗦𝗸𝘆𝗦𝗮𝗳𝗲" data product. By translating complex historical data into a simple, interpretable, and highly accurate predictive tool, we have provided a tangible solution that allows the aviation industry to move from a reactive to a proactive stance, ultimately contributing to a safer and more resilient future for air travel. 

# SkySafe - 𝑺𝒆𝒆 𝒃𝒆𝒚𝒐𝒏𝒅 𝒕𝒉𝒆 𝒊𝒎𝒑𝒂𝒄𝒕...

1. 𝗦𝘂𝗿𝘃𝗶𝘃𝗮𝗹 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁
   
   ![WhatsApp Image 2026-03-29 at 10 41 15](https://github.com/user-attachments/assets/95fa8eb9-099d-4f23-b380-f77199567ee6)

   ![WhatsApp Image 2026-03-29 at 10 41 20](https://github.com/user-attachments/assets/825e7a35-172d-4dda-bd15-f6ec529ee268)

2. 𝗜𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝘃𝗲 𝗗𝗮𝘀𝗵𝗯𝗼𝗮𝗿𝗱

   ![WhatsApp Image 2026-03-29 at 10 41 15 (2)](https://github.com/user-attachments/assets/bc99419e-2c31-4ba2-a15c-0f925c40afdc)

   ![WhatsApp Image 2026-03-29 at 10 41 16](https://github.com/user-attachments/assets/f77f37ab-9e18-4566-9a3e-ddfab76e2021)



