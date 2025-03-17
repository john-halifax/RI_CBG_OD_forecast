
Percent_OD_captured<-function(preds,yactual,
                                    action_percent){
  number_cbgs<-round(length(preds)*action_percent/100)
  values<-arrange(as.data.frame(cbind(preds,yactual)),desc(preds))
  in_specified_percent<-values[1:number_cbgs,]
  prop_overdoses_captured<-sum(in_specified_percent[,2])/sum(yactual)*100
  prioritized_cbgs<-rownames(in_specified_percent)
  return(list(Percent_total_OD_captured=prop_overdoses_captured,
              Prioritized_units=prioritized_cbgs))
}

Mean_squared_error<-function(preds,yactual){
  mse<-mean((yactual-preds)^2)
  return(mse)
}