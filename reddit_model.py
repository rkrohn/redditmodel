import load_model_data
import cascades

#driver for all the other things


code = "crypto"

print("Processing", code)

#posts, comments = load_reddit_data(code)
#cascades.build_cascades(posts, comments, code)
cascades.build_cascades(code)

#load_exogenous_data(code)
