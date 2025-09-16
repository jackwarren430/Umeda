 
class object_tree():
    def __init__(self):
        self.head
        self.main_label

    def print():
        # print into a GPT-friendly version so it can understand the structure
        return

class object():
    def __init__(self):
        self.parent_object
        self.sub_objects = []
        self.area = ((0.0, 0.0), (0.0, 0.0))
        self.file_path


 
def get_description(image_path):
    # pass image into ai model to get a base description

    prompt = '''Give me a full, complete description of this image. If it is one object in the sketch, label the object and give a description. 
    Be sure to include what you would expect it to do, or be able to do. For example, a piano should be able to be played, and a ball should bounce off the bottom of the image.
    If there are multiple objects, label and describe each one. Also include the interactions between them'''

    main_object_label = "piano"
    prompt_simple = f"Give me a full, complete descripton of this sketch of a {main_object_label}"
 


def create_object(image_path):

    base_description = get_description(image_path)
    objects = get_objects(image_path, description)



    
    create_object_file() # create a file for the object

    create_code()
    sub_objects = check_for_sub_objects()

    if sub_objects:
        create_object(sub_object)

    draw_connections()
