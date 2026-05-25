---
Congratulations on reaching the stage 2 of our interview process. For this stage we would ask you to complete a small technical test.
---

Scenario:  
UK Telecoms LTD. have decided to prioritise the retention of customers in the upcoming year and data science team to support that goal.
The ask from the business is to be able to better prioritise its resources and focus on customers that are more likely to place a cease and therefore leave for example by only calling a certain group of customers that are more likely to leave.

The approach how you deliver this solution is open to you to decide but you are required to develop your project using Python coding language.

All the information you need are included in this file: https://dev.azure.com/tt-insight-analytics/ds-tech-test/_git/tech-test?path=/README.md


You will have 10 minutes to present the model and findings to both technical (data science manager) and non-technical audience which will be followed by questions around your technical approach. After which some more general technical questions will be asked.

We would recommend spending no more than 3 hours as your time is valuable! If you are unable to complete the entire modelling process in that time (it is a tall ask!) prioritise your tasks as you feel appropriate and just talk us through your next steps.


Your submission needs to contain:
1. Suitably structured Git repo containing code

2. HTML, pdf or screen shot of the code

3. Presentation in PPT
----
Available to you is the following synthetic data:

[cease data](cease.csv)
Contains information for cease placed by customers planning to leave UK Telecoms LTD.

It contains;  
a unique reference for each individual customer (unique_customer_identifier)  
the date a cease was placed (cease_placed_date)  
if that cease completed and when (cease_completed_date)   
a brief descritption of the reason for placing a cease(reason_description)   
a condensed grouping of the reason descritpion (reason_description_insight)


![image](cease_example.PNG)


[customer info](customer_info.parquet)
contains general information around customers with UK Telecoms LTD such as information about their current contract status, tenure, days in or out of contract. Use duckdb, pandas read_parquet or your preferred approach to read in parquet files.

It contains;  
a unique reference for each individual customer (unique_customer_identifier)  
the month this data pertains to (datevalue)  
their current contract status whether they are in contract, out of contract, in the first few months / early contract, about to come out of contract etc (contract_status)
the number of times they have cancelled their direct debit in their most recent contract period (contract_dd_cancels)  
the number of times they have cancelled their direct debit in the last 60 days (dd_cancel_60_day)  
the number of days they have been out of contract, this can be positive or negative with a negative number representing time until they come out of contract (ooc_days)
the type of service they are recieving such as Fibre to the Cabinet (FTTC), Metallic Path Facility (MPF) or G.Fast which represent the technology their service is provided over (Technology)  
the speed that their product should offer (speed)  
the speed they're recieving (line_speed)  
the channel with which they were sold broadband such as calling in (Inbound), migrated from purchases of other providers (Migrated Customer) online orders (Online- X) (sales_channel)  
the name of the package they're currently on (crm_package_name)  
the number of days they have been with UK Telecoms LTD (tenure_days)  


![image](customer_info_example.PNG)

[call information](calls.csv)
contains information about calls a customer has made into our contact centre 
It contains;  
a unique reference for each individual customer (unique_customer_identifier)  
the date the call occured (event_date)  
the department the call was for such as calls to the retention team (Loyalty) or the customer services and billing team (CS&B) (call_type)  
the length of the call in seconds (talk_time_seconds)  
the length of time on the call where the customer was placed on hold in seconds (hold_time_seconds)  

![image](incoming_calls_example.PNG)


[usage data](usage.parquet)
broadband usage for customers both upload and download this is a parquet file due to its size so is not human readable. Use duckdb, pandas read_parquet or your preferred approach to read in parquet files.

It contains;  
a unique reference for each individual customer (unique_customer_identifier)  
the date the usage was recorded for (calendar_date)  
usage in megabytes that the user downloaded (usage_download_mbs)  
usage in megabytes the user uploaded (usage_upload_mbs)  


![image](bb_usage_example.PNG)
