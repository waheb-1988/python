### matplotlib
# Print the last item from year and pop
print(year[-1])
print(pop[-1])


# Import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
# Make a line plot: year on the x-axis, pop on the y-axis

plt.plot(year,pop)
# Display the plot with plt.show()
plt.show()

# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap, life_exp)

# Display the plot
plt.show()


#### Dectionnary

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger= countries.index('germany')

# Use ind_ger to print out capital of Germany

print(capitals[ind_ger])
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin','norway':'oslo'}

# Print europe
print(europe)


# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe

europe.keys()
# Print out value that belongs to key 'norway'

print(europe['norway'])