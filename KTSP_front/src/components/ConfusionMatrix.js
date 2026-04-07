import React,{useEffect,useRef} from "react";
import Chart from "chart.js/auto";
import{ Chart as ChartJS,CategoryScale ,LinearScale,BarElement,Title,Tooltip} from 'chart.js';
import{Bar} from 'react-chartjs-2';

ChartJS.register(CategoryScale,LinearScale,BarElement,Title,Tooltip);

/**
function  ConfusionMartix({matrix}){
    const data={
        labels:['Pred 0','Pred 1'],
        datasets:[
            {
                label:'Actual 0',
                data:matrix[0],
                backgroundColor: ['rgba(75,192,192,0.6)', 'rgba(255,99,132,0.6)'],
            },
            {
                label:'Actual 1',
                data:matrix[1],
                backgroundColor: ['rgba(54,162,235,0.6)', 'rgba(255,206,86,0.6)'],
            },
        ],
    };
  return <Bar data={data}> </Bar>
};
**/

const ConfusionMatrix=({data})=> {
    const chartRef = useRef(null);

    useEffect(() => {
        if (!data) return;
        const ct = chartRef.current.getContext("2d");
        new Chart(
            ct, {
                data: {

                    labels: ["Class 0", "Class 1"],
                    type: "heatmap",
                    datasets: [{

                        data: [
                            {x: 0, y: 0, v: data[0][0]},
                            {x: 1, y: 0, v: data[0][1]},
                            {x: 0, y: 1, v: data[1][0]},
                            {x: 0, y: 0, v: data[1][1]},
                        ],
                        backgroundColor: "rgba(255,0,0,0.5)"
                    }]
                }
            });
    }, [data]);


    return <canvas ref={chartRef}></canvas>;
}