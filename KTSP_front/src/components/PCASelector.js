import React from "react";
function PCASelector({usePCA,setUsePCA}){
    return(
        <div>
            <label>
                <input type ="checkbox"
                       checked={usePCA}
                       onChange={(e)=>setUsePCA(e.target.checked)}/>
                    Use PCA


            </label>
        </div>
    );
}

export default PCASelector;