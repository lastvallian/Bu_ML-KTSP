import React from "react";
import {Bar,Line} from "react-chartjs-2";
import{Chart as ChartJS,CategoryScale,LinearScale,BarElement,
PointElement,LineElement,Title,Tooltip} from "chart.js";

export const ROCChart=({data})=>
{
        //Prevention check for fpr
        if (!data || !data.fpr || !data.tpr){
        return(<div style={{ padding: "10px", color: "#888", border: "1px dashed #ccc" }} >
            Data is not completely,it can not render Roc Curve!
        </div>);
    }
      try{
            //transform data for fpr array and map
          const chartData=data.fpr.map((val,index)=>({
              fpr:val,
              tpr:data.tpr[index]
          }));
           return(
               <div style={{width: '100%', height: '300px', background: '#f9f9f9', padding: '10px'}}>
                   <p style={{ fontSize: '12px', textAlign: 'center' }}>
                       ROC cur has ready:(AUC:{data.auc?.toFixed(3)||'N/A'})
                   </p>
                   <div style={{ borderLeft: '2px solid black', borderBottom: '2px solid black', height: '200px', position: 'relative' }}>
                       <small style={{ position: 'absolute', bottom: '-20px' }}>FPR</small>
                       <small style={{ position: 'absolute', left: '-30px', top: '0' }}>TPR</small>
                   </div>
               </div>
           )}
      catch (error) {
              console.error(" Roc Chart error:", error);
              return <div>render Roc Chart error</div>;
          }


};

export const ConfusionMatrix=({cm})=>{
    if(!cm||!Array.isArray(cm)||cm.length===0){
        return <div style={{ color: "#888" }}>No confusion matrix!</div>;
    }

    const flatData=cm.flat();
    const maxVal=Math.max(...flatData,1);
    return(
        <div style={{ display: "inline-block" }}>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${cm.length}, 60px)` }}>
                {cm.map((row,i)=>
                  row.map((val,j)=>(
                    <div key={`${i}-${j}`} style={{
                width: "60px",
                height: "60px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: `rgba(0, 123, 255, ${val / maxVal})`,
                color: val / maxVal > 0.5 ? "white" : "black",
                border: "1px solid #ddd"
            }}>
                {val}
            </div>
            ))
            )}
            </div>
        </div>
    );


}
