import base64
import os
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def png_to_base64(img):
    """Convert a PNG image to a Base64 string."""
    with open(img, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def save_to_html(demes, pca_base64, map_base64, html_file, html_table, prefix, density, model, pops, html_table_masked, demes_masked, density_masked, model_masked):
    """Save the Base64 string as an image in an HTML file."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=0.8">
        <title>{prefix} Report</title>
        <style>
            body {{
                background-color: #515d85; /* Background color */
                font-family: Arial, sans-serif; /* Font style */
                color: white; /* Set text color to bone white */
                margin: 0;
                padding: 0;
            }}
            h1, h2, p {{
                color: white;
            
            }}
            h1 {{
                margin-left: 350px; /* Shift the title over to the right to avoid the About section */
                font-size: 2em;
            }}
            /* Flex container for the items */
            .container {{
                display: flex;
                margin-left: 330px; /* Push the content to the right to avoid overlap with About section */
                justify-content: space-around;
                flex-wrap: wrap; /* Allow wrapping of items if needed */
                gap: 20px; /* Add some space between the items */
            }}
            /* Flex item style (table and images) */
            .item {{
                flex: 1 1 45%; /* Allow items to grow and shrink, with a base width of 45% */
                margin: 20px;
                box-sizing: border-box; /* Include padding in width calculation */
            }}
            table {{
                border-collapse: collapse;
                width: 100%; /* Make the table take full width of its container */
                border: 4px solid #b8b9be; /* Very dark brown border */
                border-radius: 4px;
                background-color: #FFFFFF;
                color: black; 
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
                color: black; 
            }}
            th {{
                background-color: #FFFFFF;
            }}
            img {{
                max-width: 100%; /* Make sure the image scales to the container width */
                height: auto; /* Maintain aspect ratio */
                border: 4px solid #b8b9be; /* Very dark brown border */
                border-radius: 4px;
            }}
            iframe {{
                width: 100%; /* Full width for the iframe */
                height: 600px; /* Set a height for the iframe */
                border: 4px solid #b8b9be; /* Very dark brown border */
                border-radius: 4px;
            }}
            /* Fixed About Section on the Left */
            .about-section {{
                position: fixed;
                top: 0;
                left: 0;
                width: 300px; /* Set a fixed width for the About section */
                height: 100%; /* Full height of the viewport */
                padding: 20px;
                background-color: #191036; /* Dark background */
                border-radius: 0 10px 10px 0; /* Rounded right corners */
                box-shadow: 2px 0 8px rgba(0, 0, 0, 0.5); /* Subtle shadow */
                z-index: 1000; /* Make sure it's always on top */
                color: white;
                overflow-y: auto; /* Allow scrolling if the content is long */
            }}
            .about-section h3 {{
                margin-top: 0;
            }}
            .about-section p {{
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{prefix} GADMA2 Demographic Inference</h1>
        
        <div class="container">
            <div class="item">
                <h2>Parameter Summary for Best Five Log-Likelihood Models</h2>
                {html_table}
            </div>
            <div class="item">
                <h2>Log-likelihood Density Plot</h2>
                <img class="img" src="data:image/png;base64,{density}" alt="Log-Likelihood Density"/>
            </div>
        </div>

        <div class="container">
            <div class="item">
                <h2>Best Log-Likelihood Model:</h2>
                <img class="img" src="data:image/png;base64,{demes}" alt="Best Demes Model Plot"/>
            </div>
            <div class="item">
                <h2>Best Model Fit:</h2>
                <img class="img" src="data:image/png;base64,{model}" alt="Model Fit"/>
            </div>
        </div>

        <div class="container">
            <div class="item">
                <h2>Parameter Summary for Best Five Masked Log-Likelihood Models</h2>
                {html_table_masked}
            </div>
            <div class="item">
                <h2>Log-likelihood Density Plot Masked</h2>
                <img class="img" src="data:image/png;base64,{density_masked}" alt="Log-Likelihood Density"/>
            </div>
        </div>

        <div class="container">
            <div class="item">
                <h2>Best Log-Likelihood Masked Model:</h2>
                <img class="img" src="data:image/png;base64,{demes_masked}" alt="Best Demes Model Plot"/>
            </div>
            <div class="item">
                <h2>Best Masked Model Fit:</h2>
                <img class="img" src="data:image/png;base64,{model_masked}" alt="Model Fit"/>
            </div>
        </div>

        <div class="container">
            <div class="item">
                <h2>Population Definitions:</h2>
                <iframe src="data:text/html;base64,{map_base64}"></iframe>
            </div>
            <div class="item">
                <h2>PCA:</h2>
                <iframe src="data:text/html;base64,{pca_base64}"></iframe>
            </div>
        </div>

        <!-- About Section -->
        <div class="about-section">
            <h3>About</h3>
            <p>This is a demographic inference report generated for {prefix}. {len(pops)} populations are analyzed: {str(pops).replace("[","").replace("]", "").replace("\'", "")}. More info...</p>
        </div>
    </body>
    </html>
    """

    with open(html_file, "w") as file:
        file.write(html_content)

def map_populations(coords_input, popfile, pops, colors):
    coords_file = pd.read_table(coords_input)
    coords_file.columns = ('#IID', 'Longitude', 'Latitude') # Must be switched sometimes if coords file is backwards
    pop_samples = pd.read_csv(popfile, sep = ' ')
    pop_samples.columns = ('#IID', 'Population')
    filtered_coords = coords_file.merge(pop_samples, on='#IID')

    # Create map center using average of coords
    mapping = folium.Map(location=[filtered_coords['Latitude'].mean(), filtered_coords['Longitude'].mean()], zoom_start=6)

    for index,row in filtered_coords.iterrows():
        pop_color = colors[pops.index(row['Population'])]
        folium.CircleMarker(location = [row['Latitude'], row['Longitude']], radius=10, color=pop_color, fill=True, fill_color=pop_color, fill_opacity=0.8, popup=f"{row['#IID']}<br>Coordinates: {row['Latitude']}, {row['Longitude']}").add_to(mapping)


    # Create population labels
    for pop in pops:
        folium.Marker(
        location=[(filtered_coords.loc[filtered_coords['Population'] == pop])['Latitude'].mean(), (filtered_coords.loc[filtered_coords['Population'] == pop])['Longitude'].mean()], 
        icon=folium.DivIcon(html=f"""
            <div style="font-size: 26px; text-shadow: 1px 1px 1px #000; color: {colors[pops.index(pop)]};">
                {pop}
            </div>
        """)).add_to(mapping)

    # Save to an HTML string
    mapped = BytesIO()
    mapping.save(mapped, close_file=False)

    # Encode the map HTML as Base64
    mapped.seek(0)  # Move to the start of the BytesIO object
    map_base64 = base64.b64encode(mapped.read()).decode()
    return map_base64


def plot_pca(eigenvec, eigenval, popfile, pops, colors):
    pop_samples = pd.read_csv(popfile, sep = ' ')
    pop_samples.columns = ('IID', 'Population')
    
    eigenvec = pd.read_csv(eigenvec, sep = '\t')
    samples = eigenvec.merge(pop_samples, on='IID')

    samples['Color'] = None
    pca = go.Figure()

    # Plot PCA
    for index,row in samples.iterrows():
        pca.add_trace(go.Scatter(x=[row['PC1']], y=[row['PC2']], mode='markers',
        marker=dict(size=10, color=colors[pops.index(row['Population'])], line=dict(
                width=1,  # Outline width
                color='black'  # Outline color
            )), name = '', hoverinfo='none',  # Do not show default hover info
        hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>',  # Use the hover template
        text=[row['IID']]  # Wrap row['IID'] in a list
    ))

    # Update axis titles
    with open(eigenval, 'r') as variance: 
        variance_PC1 = float(variance.readline())
        variance_PC2 = float(variance.readline())
    pca.update_layout(
        title='Genomic PCA',
        xaxis_title=f'PC1: {round(variance_PC1, 2)}% Variance',
        yaxis_title=f'PC2: {round(variance_PC2, 2)}% Variance',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
        gridcolor='lightgray',
        zerolinecolor='lightgray'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        )
        )

    # Convert figure to HTML string
    html_bytes = pca.to_html(include_plotlyjs='cdn', full_html=False)
    
    # Encode the HTML to Base64
    encoded_html = base64.b64encode(html_bytes.encode('utf-8')).decode('utf-8')
    return encoded_html


def plot_likelihood(parameters, pops, density, intermediates, labels):
    params = pd.read_csv(parameters)

    plt.figure(figsize=(10, 6))

    # Generate a grid of points and compute the density
    kde = sns.kdeplot(data=params, x=f'{pops[0]}->{pops[1]}', y='tsp1 (years)', fill=True, thresh=0, levels=50)

    # Get the axes and set the limits
    ax = plt.gca()
    xlim = ax.get_xlim() # Stupid
    ylim = ax.get_ylim()
        # Create a scatter plot to represent 'log-likelihood' as hue
    scatter = plt.scatter(
        params[f'{pops[0]}->{pops[1]}'],
        params['tsp1 (years)'],
        c=params['log-likelihood'],  # Use log-likelihood for color
        cmap='viridis',          # Choose your colormap
        alpha=0.5,              # Adjust transparency
        edgecolor='w'           # White outline for points
    )

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log-Likelihood')

    # Find the index of the maximum log-likelihood
    max_ll_index = params['log-likelihood'].idxmax()
    max_ll_point = params.iloc[max_ll_index]

    # Plot the point with the highest log-likelihood as 'X'
    plt.scatter(
        max_ll_point[f'{pops[0]}->{pops[1]}'],
        max_ll_point['tsp1 (years)'],
        color='white',  # Color of the 'X'
        s=70,        # Size of the 'X'
        marker='x'    # Marker style
    )
    if intermediates: # Plot intermediate models
        runs = pd.read_csv(intermediates)

        # Iterate over independent runs and plot intermediate models from stored lists of params
        for index, row in runs.iterrows():
            run = row['run']
            exes = []
            whys = []
            for model in eval(row['intermediates']):
                y = model[labels.index("tsp1 (years)")]
                x = model[labels.index(f"{pops[0]}->{pops[1]}")]
                exes.append(x)
                whys.append(y)
            current_run = params.loc[params['Run'] == run]
            whys.append(float((current_run['tsp1 (years)']).iloc[0]))
            exes.append(float((current_run[f'{pops[0]}->{pops[1]}']).iloc[0]))
            
            plt.plot(exes, whys, linestyle='-',linewidth=1.1, alpha=0.5)  # Line plot
            
            # ATTACH LINE TO CURRENT RUN ?
            #current_run = params.loc[params['Run'] == run]
            #current_run['tsp1 (years)'], current_run[f'{pops[0]}->{pops[1]}']
            # Save the values of the relevant columns from that row
            #print(current_run['tsp1 (years)'], current_run[f'{pops[0]}->{pops[1]}'])
            #plt.plot(current_run['tsp1 (years)'], current_run[f'{pops[0]}->{pops[1]}'], label=f'Line {index + 1}')  # Plot final point of line plot


    # Add titles and labels
    #plt.title('2D Density Plot of First Split and Singular Migration Rate with Log-Likelihood Hue')
    plt.xlabel(f'{pops[0]}->{pops[1]}')
    plt.ylabel('tsp1 (years)')

    # Save the plot
    plt.savefig(density, dpi=300, bbox_inches='tight')

    return png_to_base64(density)


def main():
    from snakemake.script import snakemake
    best_demes = snakemake.input['best_plot']
    best_demes_masked = snakemake.input['best_plot_masked']
    prefix = snakemake.params['prefix']
    pops = snakemake.params['pops'].split('/')
    refGenome = snakemake.params['refGenome']
    html_file = snakemake.output['html']
    parameters = snakemake.input['estimates']
    parameters_masked = snakemake.input['estimates_masked']
    coords_input = snakemake.input['coords']
    popfile = snakemake.input['popfile']
    model_fit = snakemake.input['model_fit']
    model_fit_masked = snakemake.input['model_fit_masked']
    eigenvec = snakemake.input['eigenvec_filtered']
    eigenval = snakemake.input['eigenval_filtered']
    colors = ['rebeccapurple', 'steelblue', 'seagreen']
    density = snakemake.output['density']
    density_masked = snakemake.output['density_masked']
    intermediates_unmasked = snakemake.input['intermediate_models']
    intermediates_masked = snakemake.input['intermediate_models_masked'] # horrible fix everything

    # Plot PCA
    pca_base64 = plot_pca(eigenvec, eigenval, popfile, pops, colors)
    
    # Create interactive map
    if coords_input != []:
        map_base64 = map_populations(coords_input, popfile, pops, colors)
    
    # Alter params representation
    params = pd.read_csv(parameters)
    params.drop('Theta', axis=1, inplace=True) #?
    for col in params.columns:
        if '->' in col:
            params[col] = params[col].apply(lambda x: '0' if x == 0 else (f"{x:.2e}" if x < 1e-3 else x))
        elif 'log' in col:
            params[col] = params[col].round(decimals=1)
            params[col] = params[col].apply(lambda x: f"{x:,.0f}")
        else:
            params[col] = params[col].astype(float)
            params[col] = params[col].round(decimals=0)
            params[col] = params[col].apply(lambda x: f"{x:,.0f}")

    params_masked = pd.read_csv(parameters_masked)
    params_masked.drop('Theta', axis=1, inplace=True) #?
    for col in params_masked.columns:
        if '->' in col:
            params_masked[col] = params_masked[col].apply(lambda x: '0' if x == 0 else (f"{x:.2e}" if x < 1e-3 else x))
        elif 'log' in col:
            params_masked[col] = params_masked[col].round(decimals=1)
            params_masked[col] = params_masked[col].apply(lambda x: f"{x:,.0f}")
        else:
            params_masked[col] = params_masked[col].astype(float)
            params_masked[col] = params_masked[col].round(decimals=0)
            params_masked[col] = params_masked[col].apply(lambda x: f"{x:,.0f}")


    demes = png_to_base64(best_demes)
    demes_masked = png_to_base64(best_demes_masked)
    params_copy = params.copy()
    params_copy.drop('log-likelihood', axis=1, inplace=True)#?
    params_copy_masked = params_masked.copy()
    params_copy_masked.drop('log-likelihood', axis=1, inplace=True)#?
    
    # Plot LL density and intermediate models
    labels = params_copy.columns.tolist()
    density = plot_likelihood(parameters, pops, density, intermediates_unmasked, labels)
    labels = params_copy_masked.columns.tolist()
    density_masked = plot_likelihood(parameters_masked, pops, density_masked, intermediates_masked, labels)

    model = png_to_base64(model_fit)
    model_masked = png_to_base64(model_fit_masked)

    params.drop('Run', axis=1, inplace=True)
    params_masked.drop('Run', axis=1, inplace=True)
    html_table = params.to_html(index=False, header=True)
    html_table_masked = params_masked.to_html(index=False, header=True)
    save_to_html(demes, pca_base64, map_base64, html_file, html_table, prefix, density, model, pops, html_table_masked, demes_masked, density_masked, model_masked)


if __name__ == "__main__":
    main()