* import the csv dataset *
import delimited "/Users/cynthiax/Desktop/ECON 1660/PS6/demand_monopoly.csv"

* creating log D(p) and log (p) *
* log D(p)
gen log_s = log(s) 
* log (p)
gen log_p = log(p)

* performing the regression *
regress log_s log_p
