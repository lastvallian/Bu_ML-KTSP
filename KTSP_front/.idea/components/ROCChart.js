import React,{useEffect,useRef} from "react";
import Chart from "chart.js/auto";

const ROCChart=({data})=>{
    const chartRef=useRef(null);
    useEffect(() => {
        if(!data) return;

        const ct=chartRef.current.getContext("2d");
        new Chart(ct,{
            type:"line",
            data:{
               labels:data.fpr,
               datasets:[{
                   label:`ROC Curve (AUC=${data.auc.toFixed(2)})`,
                   data:data.tpr,
                   borderColor:"blue",
                   fill:false,
               },
               ],
            },
            options:{
                scales:{
                    x:{title:{display:true,text:"False Positive Rate"}},
                    y:{title:{display:true,text:"False Positive Rate"}}
                },
            },
        });
    }, [data]);
    return <canvas ref={chartRef}></canvas>;
};
export default ROCChart;