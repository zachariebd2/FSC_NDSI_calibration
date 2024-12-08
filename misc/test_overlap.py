import sys
import os
import errno
import re

import snowcover


#initialize
SNW = snowcover.snowcover()  #todo: add "data" as input



do_PLEIADES = False
do_SPOT = True


#PLEIADES##############################################################
if do_PLEIADES :
    print("\nPLEIADES")
    #parameters
    SNW.setEpsgFSC("")
    SNW.setSourceFSC("PLEIADES")
    SNW.setValues(False,[1],[3],[2])
    SNW.setResamplingType("average")
    
    
    #search tiles
    tiles = SNW.searchTiles()
    
    #search inputs
    #Lrasters = SNW.searchRasters(["T31TCH"])
    Lrasters = SNW.searchRasters(tiles)
    
    
    print("\n nb of FSC inputs = " + str(len(Lrasters)))
    for dateFSC in Lrasters :
        f_FSC = Lrasters[dateFSC][0]
        l_L2A = Lrasters[dateFSC][1]

        print("\nDATE = " + str(dateFSC) + "\n     FSC = " + str(f_FSC) + "\n     nb of L2A tiles = " + str(len(l_L2A)) )
        
        for tile , L2A  in l_L2A : 
            print("     " + tile + " : " + L2A)
            
    
    #processing
    OK = SNW.doProcessing()
    
    #plot
    if OK :
        OK = SNW.PlotPeriode()
        if not OK: print("\nPeriode plot problem")
        OK = SNW.PlotEachDates()
        if not OK: print("\nEachDate plot problem")
    else : 
        print("\nCalculation problem")

#SPOT##############################################################
if do_SPOT :
    print("\nSPOT")
    #parameters
    SNW.setEpsgFSC("")
    SNW.setSourceFSC("SPOT67")
    SNW.setResamplingType("average")
    SNW.setValues(False,[2],[1],[0])
    
    #search tiles
    #tiles = SNW.searchTiles()
    
    #search inputs
    Lrasters = SNW.searchRasters(["T32TLR","T32TLQ","T31TGK","T31TGL"])
    #Lrasters = SNW.searchRasters(tiles)
    
    
    print("\n nb of FSC inputs = " + str(len(Lrasters)))
    for dateFSC in Lrasters :
        f_FSC = Lrasters[dateFSC][0]
        l_L2A = Lrasters[dateFSC][1]

        print("\nDATE = " + str(dateFSC) + "\n     FSC = " + str(f_FSC) + "\n     nb of L2A tiles = " + str(len(l_L2A)) )
        
        for tile , L2A  in l_L2A : 
            print("     " + tile + " : " + L2A)
    


    #processing
    OK = SNW.doProcessing()
    
    #plot
    if OK :
        OK = SNW.PlotPeriode()
        if not OK: print("\nPeriode plot problem")
        OK = SNW.PlotEachDates()
        if not OK: print("\nEachDate plot problem")
    else : 
        print("\nCalculation problem")        
    
