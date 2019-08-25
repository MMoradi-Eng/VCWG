from UWG import UWG

# Define the .epw, .uwg filenames to create an UWG object.
# UWG will look for the .epw file in the UWG/resources/epw folder,
# and the .uwg file in the UWG/resources/parameters folder.
epw_filename = "USA_AZ_Tucson.722740_TMY2.epw"      # EPW file name
# epw_filename = "Guelph.epw"
param_filename = "Input_Parameters.uwg"         # .uwg file name

# Initialize the UWG object and run the simulation
uwg = UWG(epw_filename, param_filename,'','','','',1)
uwg.run()
