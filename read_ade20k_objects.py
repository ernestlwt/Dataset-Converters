
OBJECTS_FILE_LOCATION="/home/ernestlwt/Downloads/ernestlwt_735a2f6a/ADE20K_2021_17_01/objects.txt"
PROCESSED_OBJECTS_FILE_LOCATION="/home/ernestlwt/Downloads/ernestlwt_735a2f6a/ADE20K_2021_17_01/objects.csv"

objects_file = open(OBJECTS_FILE_LOCATION, "r")
objects_csv_file = open(PROCESSED_OBJECTS_FILE_LOCATION, "w")

TO_SCAN_KEYWORD = True
KEYWORDS = ['window', 'door']

objects_content = []
for line in objects_file:
    raw_elements = line.rstrip().split('\t')
    processed_elements = []
    for ele in raw_elements:
        processed_elements.append('"' + ele + '"')

    if any(word in processed_elements[0] for word in KEYWORDS):
        objects_csv_file.write(','.join(processed_elements))
        objects_csv_file.write('\n')
objects_file.close()
objects_csv_file.close()


REQUIRED_INDEX = [
    3055, # WINDOWPANE
    783, # DOUBLE DOOR
    1747, # GLASS PANE
    778, # DOOR FRAME
    2251, # SCREEN DOOR
    2439, # SLIDING DOOR
    754, # DISPLAY WINDOW, SHOP WINDOW
    851, # ELEVATOR DOOR
    1141, # GRILLE DOOR
    995, # FOLDING DOOR
    782, # DORMER WINDOW
    2765, # TICKET WINDOW
    852, # ELEVATOR DOORS (SIMILAR TO 851?)
    2286, # SECURITY DOOR
    3054, # WINDOW SCARF (not window but can be used to find windows?)
    2358, # SHOWER DOOR
    2164, # ROSE WINDOW
    2287, # SECURITY DOOR FRAME
    3050, # WINDOW
    774, # DOOR
    2346, #SHOP WINDOW
    776, # DOOR FRAME (SIMILAR TO 778)
    3056, # WINDOWS (SIMILAR TO 3050)
    1062, # GARAGE DOOR
    2103, # REVOLVING DOOR
    779, # DOORS (SIMILAR TO 774)
]
