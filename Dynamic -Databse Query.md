I want to launch a new project: A Web application platform that users can create the data set by the system provided Database Information.
1.In the Page, it will show all the Available Database and their Data Table List, For Each Table, it will show all the columns.
2.If there are Data Dictionary, it can provide users all the Table and Column description for users well understand the data column meaning.
3.There is a Page, users can Table by Table to select those columns that he wants.
4.There is a page, that users can well describe what the data they wants, users can one by one to key in or just import Excel, CSV, Json,... file.
5.In the page, users can setup different data criteria for specific data object and scope.
6.When users define these data set, the system will automatically to search what the best data source or just follow users identified Table and column to generate the SQL script. Remember, you must do the Performance evaluation by Key checking, Full Table Scan checking, any performance impact must be stop or notification.
7.When SQL is generate, execute the SQL and show 100 records data to GUI for user review.If users agree, then execute the complete SQL to get all data bu user criteria.
8.The system must setup the Time out and Data Size constraint to avoid DB impact.
9.Frontend use React+Vite, Backend use the Python Flask!
