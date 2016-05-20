# Function to remove unnecessary parts of input recipe data
def filterstring(str):
    str = ','.join([filterstring_helper( x ) for x in str.split(',')])
    return str

def filterstring_helper(str):
    brands_list = ['adobo', 'azteca', 'bertolli', 'bestfood', 'bettycrocker', 'bisquick', 'bragg', 'breakstones',
                   'breyers', 'campbells', 'cavenders', 'cholula', 'classico', 'colmans', 'coxs', 'crisco',
                   'crystalfarms', 'curryguy', 'delallo', 'diamondcrystal', 'earthbalance', 'estancia',
                   'fosterfarms', 'franks', 'gebhardt', 'goldmedal', 'goodseasons', 'gourmetgarden', 'goya',
                   'greengiant', 'heath', 'heinz', 'hellmann', 'herdez', 'hiddenvalley', 'hogue', 'hollandhouse',
                   'homeoriginals', 'honeysucklewhite', 'imperialsugar', 'jimmy dean', 'johnsonville', 'karo',
                   'kerrygold', 'kewpie', 'kikkoman', 'kimcrawford', 'klondikerose', 'knorr', 'knox', 'knudsen',
                   'kraft', 'kroger', 'lavictoria', 'landolakes', 'leaperrins', 'lipton', 'louisiana', 'mms',
                   'maeploy', 'martha', 'massaman', 'mazola', 'mccormick', 'meyer', 'mission', 'mizkanoigatsuo',
                   'morton', 'nakano', 'nido', 'nielsenmassey', 'oldelpaso', 'oreo', 'ortega', 'oscarmayer',
                   'pam', 'panetini', 'pepperidgefarm', 'pillsbury', 'progresso', 'ragu', 'redgold', 'royal',
                   'saffronroad', 'sanmarzano', 'sargento', 'smartbalance', 'smithfield', 'soyvay', 'spiceislands',
                   'spring!', 'stonefire', 'success', 'swanson', 'tabasco', 'tacobell', 'tapatio', 'tuttorosso',
                   'unclebens', 'wesson', 'wishbone', 'zatarains']

    adjectives_list = ['bonein', 'boneless', 'brinecured', 'chop', 'cookanddrain', 'crumbles', 'crush', 'cutinto',
                       'drain', 'extrafirm', 'fatfree', 'fine', 'finely', 'firmlypacked', 'fresh', 'frozen',
                       'fullycooked', 'glutenfree', 'grassfed', 'ground', 'homemade', 'lean', 'lesssodium', 'lowfat',
                       'lowsodium', 'lower sodium', 'lowsodium', 'nonhydrogenated', 'oz', 'partskim', 'peelanddevein',
                       'peeled', 'reducedfat', 'reducedsodium', 'refrigerated', 'rinseandpatdry', 'saltfree',
                       'servingpieces', 'skinless', 'sliced', 'slimcut', 'soften', 'splitandtoasted', 'storebought',
                       'thaw', 'thawedandsqueezeddry', 'uncook', 'undrain']

    # copyright, registered, trademark, acute a, circumflex a, manada a, section, cent
    symbols_list = [u"\u00a9", u"\u00ae", u"\u2122", u"\u00e1", u"\u00e2", u"\u00e3", u"\u00a7", u"\u00a2"]
    # Convert string to lower case
    str = str.lower()
    # Remove any symbols
    for s in symbols_list:
        new_string = str.replace(s, "")
        str = new_string
    # Remove whitespace and non-alpha characters
    str = "".join([i for i in str if i.isalpha()])
    # Remove the brand names
    for b in brands_list:
        new_string = str.replace(b, "")
        str = new_string
    # Remove any adjectives that are in the list
    for a in adjectives_list:
        new_string = str.replace(a, "")
        str = new_string
    # Return the processed string
    return str