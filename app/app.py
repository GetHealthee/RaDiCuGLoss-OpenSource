from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from radicugloss.radicugloss import nrdcgl
from logger.logger import JSONLogger

json_logger = JSONLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'OK'}), 200


@app.route('/nrdcgl', methods=['POST'])
def nrdcgl_endpoint():
    json_logger.info('nrdcgl endpoint called')
    if not request.is_json:
        json_logger.info('Missing JSON in request')
        abort(400, description="Missing JSON in request")
    data = request.get_json()

    search_results = data['search_results']
    true_relevance_set = data['true_relevance_set']

    if not search_results or not true_relevance_set:
        json_logger.error('search_results or true_relevance_set are missing in data')
        abort(400, description="search_results or true_relevance_set are missing in data")

    if not isinstance(search_results, list) or not isinstance(true_relevance_set, dict):
        json_logger.error('search_results should be a list and true_relevance_set should be a dict')
        abort(400, description="search_results should be a list and true_relevance_set should be a dict")

    k = data.get('k')
    fp_penalty = data.get('fp_penalty', 1)
    fn_penalty = data.get('fn_penalty', 1)
    invert = data.get('invert', True)
    punish_max = data.get('punish_max', False)

    try:
        json_logger.info('Calling nrdcgl')
        result = nrdcgl(search_results,
                        true_relevance_set,
                        k,
                        fp_penalty,
                        fn_penalty,
                        invert,
                        punish_max)
    except Exception as e:
        json_logger.error('Error in nrdcgl: %s', str(e))
        abort(500, description=str(e))
        raise e

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5678)
