Task 1:





repeat\_purchase\_flag column is the label.



Choice:  repeat\_purchase\_flag is considered as label. because label always choose binary values like 0/1. if the customer made a repeat purchase within 30 days then yes(1). if not then no(1). 



avg\_order\_value would introduce data leakage.



Choice: avg\_order\_value causes data leakage. because, average can only be calculated after the order value. if we cannot calculate the order value then average column will not appears. It was a future column. If future column is added as a feature then it causes data leakage. 





Task:2



first 2 steps in ML work flow:



Frame:   Define a target ( without a  clear problem definition, you risk building a model that doesn’t align with real-world needs.)





Audit: ethics \& PII (PII auditing is mandatory because it protects people, ensures compliance, reduces risk, and improves model quality.)



