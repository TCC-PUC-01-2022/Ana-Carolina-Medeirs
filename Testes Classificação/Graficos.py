# Import required libraries
import pandas as pd
  
# Create a sample dataframe
df = pd.DataFrame({'Number of Males': [10, 15, 25, 14],
                   'Number of Females': [20, 25, 15, 10]},
                  index=['Italian', 'Indian', 'Mexican', 'Chinese'])
  
# Plot stacked horizontal bar chart
df.plot(kind="barh", title="Gender wise Cuisine preference chart",
        color={"green", "pink"})