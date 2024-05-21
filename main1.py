import mercantile, requests, os
from vt2geojson.tools import vt_bytes_to_geojson

# Ensure these directories exist or the script will not run correctly
if not os.path.exists('output'):
    os.makedirs('output')

# Define an empty GeoJSON structure to collect features
output_geojson = {
    "type": "FeatureCollection",
    "features": []
}

# Configuration for Mapillary API
tile_coverage = 'mly1_public'  # Mapillary Vector Tile coverage type
tile_layer = 'image'  # Appropriate layer based on the desired data
access_token = 'MLY|7648402008531112|ac74d71a8d8cb83e1b35d9c553438f5b'  # Replace with your Mapillary access token

# Define the geographical bounding box
# west, south, east, north = [-80.13423442840576,25.77376933762778,-80.1264238357544,25.788608487732198]
# west, south, east, north = [-83.468632963415, 42.4712742373734, -83.41732101680651, 42.49869234613597]  # Example coordinates
# west, south, east, north = [-83.5532633226741, 42.43302319281859, 83.34810089153233, 42.51656225087703]

# west, south, east, north = [11.477553225690132, 48.521238923077085, 11.485313186246147, 48.52532145934347]
#
# west, south, east, north = [11.51422, 48.16665, 11.55642, 48.19522]
#
west, south, east, north = [11.51422, 48.16665, 11.53425, 48.19522]
#
# west, south, east, north = [11.40282, 48.02075, 11.86392, 48.38447]

# Retrieve tiles at zoom level 14
tiles = list(mercantile.tiles(west, south, east, north, 14))

# Loop through tiles to fetch and process vector tile data
for tile in tiles:
    tile_url = f'https://tiles.mapillary.com/maps/vtp/{tile_coverage}/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}'
    response = requests.get(tile_url)
    if response.status_code == 200:
        data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=tile_layer)

        # Process each feature within the current tile
        for feature in data['features']:
            coordinates = feature['geometry']['coordinates']
            lng, lat = coordinates[0], coordinates[1]

            # Check if feature is within the bounding box
            if west < lng < east and south < lat < north:
                sequence_id = feature['properties'].get('sequence_id')
                image_id = feature['properties'].get('id')

                if sequence_id and image_id:
                    # Ensure a directory for each sequence exists
                    sequence_path = os.path.join('output', sequence_id)
                    if not os.path.exists(sequence_path):
                        os.makedirs(sequence_path)

                    # Request image data using the image ID
                    headers = {'Authorization': f'OAuth {access_token}'}
                    image_url = f'https://graph.mapillary.com/{image_id}?fields=thumb_2048_url'
                    image_response = requests.get(image_url, headers=headers)
                    if image_response.status_code == 200:
                        image_data_url = image_response.json().get('thumb_2048_url')

                        # Download and save the image
                        if image_data_url:
                            image_data = requests.get(image_data_url, stream=True).content
                            with open(os.path.join(sequence_path, f'{image_id}.jpg'), 'wb') as file:
                                file.write(image_data)
                    else:
                        print(f"Failed to retrieve image {image_id}: {image_response.status_code}")
                else:
                    print(f"Missing 'sequence_id' or 'id' in feature properties")
    else:
        print(f"Failed to load tile at {tile.z}/{tile.x}/{tile.y}: {response.status_code}")

# Optionally, save or process the output_geojson as needed
