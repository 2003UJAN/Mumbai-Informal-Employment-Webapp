document.addEventListener("DOMContentLoaded", function () {

    let labels = statsData.map(x => x.area_name);
    let density = statsData.map(x => x.informal_employment_density);

    let chartData = [{
        x: labels,
        y: density,
        type: "bar"
    }];

    Plotly.newPlot("chart_overlap", chartData, { title: "Informal Employment Density by Area" });

});

