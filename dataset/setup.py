import os, json

data_path = 'Customization'

# read all the subfolder in the data_path
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

caption_list = {
    'backpack': [
        "A photo of a woman wearing a backpack, viewed from the back, against a cloudy sky.",
        "A woman with backpack, emphasizing the contrast against the cloudy sky.",
        "A close-up of the backpack on a rocky mountain surface.",
        "The backpack placed in a lush green forest setting.",
        "A view of the backpack under a fallen tree in a forest, rugged setting.",
        "The backpack hanging from a branch in a forest, captured in a serene and natural environment.",
    ],
    'backpack_dog': [
        "A photo of a backpack_dog perched on a windowsill with a view of city buildings in the background.",
        "A similar view of the backpack_dog on a windowsill, overlooking a cityscape with prominent buildings.",
        "A photo of a backpack_dog on a wooden table inside a well-lit lounge.",
        "The backpack_dog hung on a metal fence in a lush garden, emphasizing the natural green surroundings.",
        "A photo of the backpack_dog placed on grass.",
    ],
    'bear_plushie': [
        "A photo of a bear_plushie sitting on a large rock in an outdoor garden setting.",
        "A photo of a bear_plushie seated on a concrete ledge by a river, with trees in the background.",
        "A photo of a bear_plushie placed in the middle of an asphalt road, with visible street markings.",
        "A photo of a bear_plushie on lush green grass, with natural lighting.",
        "A photo of a bear_plushie against a brick wall, showcasing its texture and urban backdrop.",
    ],
    'berry_bowl': [
        "A photo of a berry_bowl filled with blueberries, resting on a soft white fabric.",
        "A close-up photo of a berry_bowl on a white cloth, filled to the brim with fresh blueberries.",
        "A photo of a berry_bowl on a plain white background, showcasing a heaping of juicy blueberries.",
        "An angled photo of a berry_bowl with blueberries, emphasizing the 'Bon Appetit' text on the bowl.",
        "A minimalist photo of a berry_bowl on a white fabric surface, highlighting the rich color of the blueberries.",
        "A photo of a berry_bowl, resting on a soft white fabric.",
    ],
    'can': [
        "A photo of a can on a stone ledge with a blurred waterway and buildings in the background.",
        "A photo of a can on a rocky surface.",
        "A photo of a can on a frozen surface.",
        "A photo of a can on a frozen surface with residential buildings in the distance.",
        "A photo of a can on a wooden surface indoors next to a glass of beer.",
        "A photo of a can and a glass of beer on a rock by the river, with a kayaker in the background.",
    ],
    'candle': [
        "A photo of a candle, placed on a rough stone surface, surrounded by an urban park environment.",
        "A candle set on a worn green wooden table, emphasizing its setting in an outdoor park.",
        "A candle positioned on lush green grass, highlighting the natural, vibrant greenery around it.",
        "A candle on grass in front of a large tree trunk, depicting a peaceful, wooded park scene.",
        "A candle placed against a red brick wall on a concrete surface, providing a stark urban backdrop.",
    ],
    'cat': [
        "A cat, sitting among dappled sunlight filtering through dense foliage, offering a serene garden vibe.",
        "A cat positioned on ground covered with leaves and small plants, highlighted by the soft, natural light peeking through the tree canopy.",
        "A cat captured in a natural setting with a backdrop of dark, leafy greens and patches of sunlight.",
        "A cat on the forest floor surrounded by scattered leaves and greenery.",
        "A cat set against a background of rich foliage and diffused light.",
    ],
    'cat2': [
        "A cat positioned on a light wooden floor.",
        "A cat lounging on a sleek surface, surrounded by subtle office decor elements.",
        "A cat seated on a bed with graphic bedding, framed by soft home lighting.",
        "A cat beside a vase of tulips on a modern coffee table.",
        "A cat relaxing by a window with a view of a rain-speckled window.",
    ],
    'clock': [
        "A clock held aloft, framed by a smooth, plain gray background that accentuates its outline.",
        "A clock cradled in a palm, set against a clean, uncluttered gray backdrop.",
        "A clock surrounded by scattered coffee cups and colorful paper.",
        "A clock resting on a soft white fabric.",
        "A clock resting on a textured yellow blanket.",
        "A clock lying on a soft, white sheet.",
    ],
    'colorful_sneaker': [
        "A colorful_sneaker rests on a concrete ledge beside a calm river, blending urban elements with natural water views.",
        "The colorful_sneaker is placed on a tarmac surface.",
        "Nestled in lush green grass, the colorful_sneaker stands out.",
        "the colorful_sneaker positioned on a worn green bench.",
        "the colorful_sneaker put against a brick backdrop.",
    ],
    'dog': [
        "A photo of a joyful dog with its tongue out, up close, against a blurred background of pink cherry blossoms.",
        "A portrait of a smiling dog standing on a ledge with cherry blossoms in the background.",
        "A photo of a cheerful dog on a ledge with a vibrant red background and cherry blossoms.",
        "A side view of a dog sitting on a ledge, turning its head towards the camera, with a blurred background of red and pink hues.",
        "An outdoor shot of a dog on a sandy path with green trees and a pond in the background.",
    ],
    'dog2': [
        "A photo of a dog walking down a forest path with fallen leaves and bare trees in the background.",
        "A photo of a dog resting on a moss-covered log in a dense forest setting.",
        "A photo of a dog lying on a porch with wooden planks, with parts of a white house visible in the background.",
        "A photo of a dog lying on a sandy path, surrounded by leafless trees and a diffused background.",
        "A photo of a dog resting on a concrete surface.",
        "A photo of a dog sitting on a grassy area surrounded by scattered leaves and autumn foliage.",
    ],
    'dog3': [
        "A photo of a dog in a forest with dry underbrush and scattered sunlight filtering through the trees.",
        "A photo of a dog on a mountain trail surrounded by dense green vegetation and rocky terrain.",
        "A photo of a dog by a river with smooth stones and water in the background, flanked by hiking shoes.",
        "A photo of a dog on a textured rock formation with visible sediment layers and sparse vegetation.",
        "A photo of a dog running on a rocky coastal area with calm sea water and distant cliffs in the background.",
        "A photo of a dog on a patterned comforter in a dimly lit room.",
    ],
    'dog5': [
        "A photo of a dog lying on a teal couch with textured grey blankets, near a window with white blinds and a teal wall.",
        "A photo of a dog on a grey sofa, surrounded by a dark grey and green throw pillow, under a white lamp with a black stand.",
        "A photo of a dog positioned between grey and green throw pillows on a grey couch, near a window with blinds and a dark grey wall.",
        "A photo of a dog lying on a patterned throw on a grey couch, with a decorative lamp and mirrored furniture in the background.",
        "A photo of a dog resting on a grey textured blanket over a couch, in a room with a blue-grey wall and a decorative lamp.",
    ],
    'dog6': [
        "A photo of a dog against a vibrant, monochrome orange background",
        "A photo of a dog positioned in front of a uniformly bright orange background.",
        "A photo of a dog showcased against a simple orange backdrop, focusing on its curious look and the details of its facial features.",
        "A photo of a dog with a joyful expression, set against a bold orange background that complements its lively energy and playful pose.",
        "A photo of a dog against a luminous yellow background, which accentuates its calm posture and attentive gaze.",
    ],
    'dog7': [
        "A photo of a dog with a close-up shot capturing its face with a backdrop of blurred forest greenery.",
        "A photo of a dog lying on a sandy path, surrounded by natural forest settings and trees in soft focus.",
        "A photo of a dog at the beach with a gentle surf in the background, highlighting the serene evening sky.",
        "A photo of a dog frolicking in ocean waves at sunset, with the vibrant colors of the sky reflecting on the water.",
        "A photo of a dog sitting amidst tall purple flowers, with soft focus on the lush greenery in the background.",
    ],
    'dog8': [
        "A photo of a dog navigating through a forest, surrounded by dense green bushes and scattered leaves on the ground.",
        "A photo of a dog seated on an asphalt road lined with lush trees forming a green tunnel in the background.",
        "A photo of a dog on a white bedspread in a bright room with soft natural light filtering through sheer curtains.",
        "A photo of a dog in a field of purple wildflowers, with vibrant green foliage enhancing the colorful backdrop.",
        "A photo of a dog lying on a patterned bedspread in a sunlit room, with shadows creating a warm, inviting atmosphere.",
    ],
    'duck_toy': [
        "A photo of a duck_toy on a textured playground surface, with leaves and shadows creating a playful contrast.",
        "A photo of a duck_toy on a soft blue carpet, with hints of a dark corner of a room in the background.",
        "A photo of a duck_toy on a dark grey yoga mat, with a subtle patterned texture in a home setting.",
        "A photo of a duck_toy positioned on a rocky beach with the ocean in the background under a clear sky.",
    ],
    'fancy_boot': [
        "A photo of a fancy_boot against a brick wall, positioned on a concrete sidewalk with leaves scattered around.",
        "A photo of a fancy_boot on a concrete edge by a river, with a city skyline and overcast sky in the background.",
        "A photo of a fancy_boot on an asphalt surface, framed by the muted colors of a parking area and buildings in the backdrop.",
        "A photo of a fancy_boot on a grassy surface, with trees and a park setting enhancing the naturalistic background.",
        "A photo of a fancy_boot by a tree in a lush green park.",
        "A photo of a fancy_boot on a weathered green bench, surrounded by a green park environment with trees.",
    ],
    'grey_sloth_plushie': [
        "A photo of a grey_sloth_plushie, sitting under a tree on lush green grass in a park setting.",
        "A photo of a grey_sloth_plushie, seated on a concrete edge by a river with an urban skyline and overcast sky in the background.",
        "A photo of a grey_sloth_plushie, seated on bright green grass with the expanse of a park in the background.",
        "A photo of a grey_sloth_plushie, seated on a weathered green park bench surrounded by trees and foliage.",
        "A photo of a grey_sloth_plushie, seated on an asphalt surface with scattered leaves, in an outdoor urban environment.",
    ],
    'monster_toy': [
        "A photo of a monster_toy, positioned on a windowsill with an expansive view of modern office buildings and a city park below.",
        "A photo of a monster_toy, sitting on top of a computer monitor in a brightly lit office environment with a view of workstations in the background.",
        "A photo of a monster_toy, perched on a windowsill with sunlight streaming through, highlighting a vibrant garden outside.",
        "A photo of a monster_toy, sitting on the hood of a car reflecting the clear blue sky and trees around in an urban parking area.",
        "A photo of a monster_toy, positioned on a rocky surface with sparse greenery, in a natural yet somewhat urbanized setting.",
    ],
    'pink_sunglasses': [
        "A photo of pink_sunglasses arranged on a white textured fabric, accompanied by magnolia blossoms adding a delicate and serene touch.",
        "A photo capturing pink_sunglasses on a subtle grey backdrop.",
        "A pink_sunglasses laid on a silky fabric, with a gentle curve of the glasses emphasizing the softness of the setting.",
        "A photo of pink_sunglasses on a crumpled white fabric.",
        "A pink_sunglasses on a bright white surface, accompanied by a magazine.",
        "A photo of pink_sunglasses on a bed, next to a magazine and a glass of juice.",
    ],
    'poop_emoji':[
        "A photo of a poop_emoji toy on an asphalt surface, surrounded by a neutral urban setting with small debris scattered around.",
        "a poop_emoji toy nestled in lush green grass, bringing a whimsical contrast to the natural environment.",
        "A photo of a poop_emoji toy in front of a large tree in a park, highlighting its playful character against the backdrop of a serene natural setting.",
        "A photo of a poop_emoji toy on a weathered green bench, complementing the quirky charm of the toy with the rustic outdoor setting.",
        "A photo of a poop_emoji toy against a brick wall, placed on a concrete surface, blending urban textures with a playful subject.",
    ],
    'rc_car': [
        "A photo of an rc_car, prominently placed on an elevated surface with a panoramic view of a sprawling construction site in the background during twilight.",
        "A photo of an rc_car, set against a domestic backdrop featuring patterned flooring and bookshelves filled with colorful books.",
        "A photo of an rc_car, positioned on an orange surface at a park, with roads and vehicles in the distant background and a clear sky.",
        "A photo of an rc_car, situated on a grey textured carpet inside a home.",
        "A photo of an rc_car, located on an asphalt road surrounded by lush greenery and fallen leaves.",
    ],
    "red_cartoon": [
        "A photo of a red_cartoon, positioned in front of a plain white background.",
        "a red_cartoon, set against a stark white backdrop.",
        "A photo of a red_cartoon, with a minor decoration symbol on its head, presented on a pure white background.",
        "a red_cartoon, displayed with its arms wide open, against an unadorned white background.",
    ],
    "robot_toy": [
        "A photo of a robot_toy positioned on a soft couch, with the grey textured fabric of the couch.",
        "A photo of a robot_toy placed on a wooden surface, highlighting the intricate patterns of the wood beneath the stark.",
        "A photo of a robot_toy outside, with vivid autumn leaves and the natural, earthy backdrop.",
        "A photo of a robot_toy on rocky terrain, surrounded by green pine needles, showcasing a natural forest setting.",
        "A photo of a robot_toy on a stone surface, with blurred autumnal forest scenery in the background.",
    ],
    'shiny_sneaker': [
        "A photo of a shiny_sneaker on a textured rock surface, with urban park elements and a large boulder in the background.",
        "A photo of a shiny_sneaker on a concrete surface beside a body of water, reflecting the light and colors of the surroundings.",
        "A photo of a shiny_sneaker on an asphalt road, against the plain, dark background.",
        "A photo of a shiny_sneaker on lush green grass.",
        "A photo of a shiny_sneaker on a worn green wooden bench.",
        "A photo of a shiny_sneaker against a brick wall on a pavement."
    ],
    'teapot': [
        "A photo of a teapot on a dark wooden table, accompanied by a white rose and an autumn leaf.",
        "A photo of a teapot on a rustic wooden bench against a weathered white wall.",
        "A photo of a teapot on a textured metallic plate, placed on a wooden surface.",
        "A photo of a teapot on a dark, moody background with soft lighting.",
        "A photo of a teapot on a textured stone pedestal beside a delicate teacup, set against a clean white background."
    ],
    'vase': [
        "A photo of a vase on a dark tabletop, accented by dried flowers and soft natural light filtering through a large window.",
        "A photo of a vase on a dark table in a rustic setting, minimalistic background.",
        "A photo of a vase against a plain white background.",
        "A photo of a vase against a plain white background.",
        "A photo of a vase on a book, against a white background, where the focus is on the vase and the accompanying floral arrangement.",
        "A photo of a vase on a minimalist background."
    ],
    'wolf_plushie': [
        "A photo of a wolf_plushie on a textured rock surface in an urban park setting, with lush greenery and a large boulder in the background.",
        "A photo of a wolf_plushie on vibrant green grass.",
        "A photo of a wolf_plushie near the base of a large tree on a grassy field.",
        "A photo of a wolf_plushie on a green wooden bench.",
        "A photo of a wolf_plushie on a brick sidewalk."
    ]
}

subject_list = ['bear_plushie', 'cat', 'cat2', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8', 'grey_sloth_plushie', 'monster_toy', 'red_cartoon', 'wolf_plushie']
def get_subject_eval_list(subject_class):
    subject_eval_list = [
        # original
        f"a photo of a {subject_class}",
        # texture change
        f"a {subject_class} made of lego",
        f"a {subject_class} in minecraft style",
        f"a {subject_class} playing a violin in sticker style",
        f"a {subject_class} gracefully leaping in origami style",
        f"a {subject_class} carved as a knight in wooden sculpture",
        # background change and custume change
        f"a photo of a {subject_class} in a wizard costume casting spells in a magical forest",
        f"a photo of a {subject_class} as an astronaut in a space suit, walking on the surface of Mars",
        # pose change
        f"a {subject_class} sleeping on a sofa",
        f"a side view of a {subject_class} in times square, looking up at the billboards",
    ]
    return subject_eval_list

object_list = ['backpack', 'backpack_dog', 'berry_bowl','can','candle', 'clock', 'colorful_sneaker', 'duck_toy', 'fancy_boot', 'pink_sunglasses', 'poop_emoji', 'rc_car', 'robot_toy', 'shiny_sneaker', 'teapot', 'vase']
def get_object_eval_list(object_class):
    object_eval_list = [
        # original
        f"a photo of a {object_class}",
        # texture change
        f"a {object_class} made of lego",
        f"a {object_class} in minecraft style",
        f"a {object_class} in origami style",
        f"a photo of cube-shaped {object_class}",
        f"a photo of {object_class} made out of lethers",
        f"a photo of a transparent {object_class}",
        f"a {object_class} featuring a design inspired by the Arsenal Football Club",
        # background
        f"a {object_class} lying on leaves in the forest",
        f"a side view of a {object_class} lying on Mars",
    ]
    return object_eval_list

setting = 'gpt_cc'
prefix_path = 'dataset/Customization/'

# traverse all the subfolders
for subfolder in subfolders:
    subfolder_name = os.path.basename(subfolder)
    if subfolder_name in caption_list:
        print(f"Processing {subfolder_name}...")
        # read all the file name if it's a image to a list in this subfolder 
        file_list = [f.name for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.jpg')]
        # sort the file list, the file will be like 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11...
        file_list.sort()
        # add the prefix to the file_list
        file_list = [os.path.join(prefix_path, subfolder_name, f) for f in file_list]
        
        # create a config dictionary
        config = {}
        config[setting] = {}
        config[setting]['caption'] = caption_list[subfolder_name]
        config[setting]['class'] = subfolder_name
        config[setting]['images'] = file_list
        if subfolder_name in subject_list:
            config[setting]['eval_prompts'] = get_subject_eval_list(subfolder_name)
        else:
            config[setting]['eval_prompts'] = get_object_eval_list(subfolder_name)
        # dump it to the config.json file
        with open(os.path.join(subfolder, 'config.json'), 'w') as f:
            json.dump(config, f)
