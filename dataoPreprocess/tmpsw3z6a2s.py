from __future__ import absolute_import, division, print_function



def create_converted_entity_factory():

  def create_converted_entity(ag__, ag_source_map__, ag_module__):

    def tf__call(self, inputs, state):
      """Gated recurrent unit (GRU) with nunits cells."""
      do_return = False
      retval_ = ag__.UndefinedReturnValue()
      ag__.converted_call(_check_rnn_cell_input_dtypes, None, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), ([inputs, state],), None)
      gate_inputs = ag__.converted_call('matmul', math_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (ag__.converted_call('concat', array_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), ([inputs, state], 1), None), self._gate_kernel), None)
      gate_inputs = ag__.converted_call('bias_add', nn_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (gate_inputs, self._gate_bias), None)
      value = ag__.converted_call('sigmoid', math_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (gate_inputs,), None)
      r, u = ag__.converted_call('split', array_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (), {'value': value, 'num_or_size_splits': 2, 'axis': 1})
      r_state = r * state
      candidate = ag__.converted_call('matmul', math_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (ag__.converted_call('concat', array_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), ([inputs, r_state], 1), None), self._candidate_kernel), None)
      candidate = ag__.converted_call('bias_add', nn_ops, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (candidate, self._candidate_bias), None)
      c = ag__.converted_call('_activation', self, ag__.ConversionOptions(recursive=True, force_conversion=False, optional_features=(), internal_convert_user_code=True), (candidate,), None)
      new_h = u * state + (1 - u) * c
      do_return = True
      retval_ = new_h, new_h
      cond = ag__.is_undefined_return(retval_)

      def get_state():
        return ()

      def set_state(_):
        pass

      def if_true():
        retval_ = None
        return retval_

      def if_false():
        return retval_
      retval_ = ag__.if_stmt(cond, if_true, if_false, get_state, set_state)
      return retval_
    tf__call.ag_source_map = ag_source_map__
    tf__call.ag_module = ag_module__
    tf__call.autograph_info__ = {}
    return tf__call
  return create_converted_entity
